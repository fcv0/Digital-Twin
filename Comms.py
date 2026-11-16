import socket
import time
import threading
import random
import math
import traceback
import numpy as np
from Utils import utils, ODCont, ODThermalCorrector, TempPIController, RingBuffer, submitFlush
from Priv import _Vars
from Predictor import RealtimeODFilter, make_od_filter, ODPipelineCompare

# OD control modes — these values are the wire protocol (OutgoingMessage[32])
# and must stay in sync with App.changeODState and the ESP32 od_mode handling.
OD_MODE_BANGBANG = 0
OD_MODE_HYST     = 1
OD_MODE_CLAMP    = 2

# Bang-bang runs for this long after OD first reaches target, then hands off to clamp.
BB_WARMUP_SECONDS = 400

class communications:
    saw = False
    def __init__(self):
        pass

    @staticmethod
    def _set_od_mode(device, port, mode):
        """Single place that changes the active OD control mode.
        Writes the wire byte and flags the packet for sending."""
        device["OutgoingMessage"][32] = mode
        device["DeviceVariables"]["SendMessage"] = True

    @staticmethod
    def _pump_state(device):
        """MotorState value for this cycle. Mixed semantics (by design):
          - Clamp (mode 2): the 0-255 pump LEVEL the controller commanded
            (d_time field, bytes 24-27; 0 = hold). This is a continuous
            proportional pump speed, not a dose time.
          - Bang-bang / hysteresis: 1 = ON (trigger byte 33 == 255), else 0.
          - Manual override: 1 if waste/fresh speed > 0 (bytes 40/42), else 0.
        Gated by the MCU's turbidostat_enabled (byte 34) auto/manual flag."""
        msg = device["OutgoingMessage"]
        if int(msg[34].item()):                       # auto OD control owns the pumps
            if int(msg[32].item()) == OD_MODE_CLAMP:
                try:
                    # Clamp to the valid digipot range so a stale/seed value in
                    # byte 24 (e.g. the bang-bang delay before ODCont's first
                    # update) can't spike the Motor plot above 255.
                    level = float(utils.unpackFloatingFromBytes(msg[24:28].tobytes()))
                    return max(0.0, min(255.0, level))  # 0-255 level
                except Exception:
                    return 0
            return 1 if int(msg[33].item()) == 255 else 0    # bang-bang / hysteresis
        return 1 if (int(msg[40].item()) > 0 or int(msg[42].item()) > 0) else 0  # manual override


    @staticmethod
    def _control_state(device):
        """(u, error, I) from the ODCont controller for this sample, logged
        every cycle alongside MotorState so the saved columns stay
        length-aligned.
          - u     : last controller output (0-255 pump level; 0 while holding)
          - error : last e = target_od - current_od (negative => dilute)
          - I     : last integral term (starts at I_initial, held until the
                    first upward setpoint crossing)
        ODCont only recomputes these in clamp mode; in other modes they hold the
        controller's last values."""
        oc = device.get("ML", {}).get("odCont")
        if oc is None:
            return 0.0, 0.0, 0.0
        return (
            float(getattr(oc, "last_u", 0.0)),
            float(getattr(oc, "last_error", 0.0)),
            float(getattr(oc, "last_I", 0.0)),
        )


    @staticmethod
    def _despike_allowed(device) -> bool:
        """False while the dilution pump is dosing (or just has been).

        A commanded dilution is a REAL, fast OD downshift — exactly what a
        despiker is built to reject. Gating it out would hold the estimate high
        while the vessel actually dilutes, and the controller would keep dosing.
        _pump_state() already reports dosing across every mode (clamp dose-ms,
        bang-bang/hyst trigger, manual override), so reuse it.

        The hold-off continues past the pump stopping: fresh media needs time to
        mix, so OD keeps falling after the dose ends.
        """
        try:
            # Manual override: a non-zero fresh/waste speed means the pump is driven.
            msg = device["OutgoingMessage"]
            if not int(msg[34].item()):                    # manual mode
                if int(msg[40].item()) > 0 or int(msg[42].item()) > 0:
                    device["_pump_last_active"] = time.time()
                    return False
            # Auto modes: the CONTROLLER stamps _pump_last_active when it actually
            # commands a dose (utils.bangbang / ODHyst / ODCont._pump). Do NOT infer
            # this from _pump_state(): in clamp mode that returns d_time, the
            # commanded dose LENGTH, which is non-zero at rest and would latch the
            # gate on forever (and disable despiking entirely).
            hold = float(_Vars.config.get("DespikeHoldoffS", 20.0))
            return (time.time() - device.get("_pump_last_active", 0.0)) > hold
        except Exception:
            return True


    @staticmethod
    def _commanded_dose_ms(device) -> float:
        """Fresh-media dose the controller commanded THIS cycle, in ms.

        Mirrors the MCU's reading of the packet (see byte_map):
          clamp    (2): bytes 24-27 is a 0-255 pump LEVEL held for the whole 1 s
                        cycle; dose-equivalent ms = 1000 * level/255 (0 = hold)
          bang-bang(0): each trigger 255 fires one d_time dose
          hyst     (1): trigger 255 = pump on for the whole 1 s cycle
          manual      : fresh speed (byte 42) non-zero = running
        Note this is the COMMAND, not _pump_state() — which reports d_time even
        at rest and so cannot tell you whether a dose actually fired.
        """
        msg = device["OutgoingMessage"]
        try:
            if not int(msg[34].item()):                       # manual override
                return 1000.0 if int(msg[42].item()) > 0 else 0.0
            mode = int(msg[32].item())
            d_time = float(utils.unpackFloatingFromBytes(msg[24:28].tobytes()))
            if mode == OD_MODE_CLAMP:
                # Proportional speed: level 0-255 held for the cycle. Scale a full
                # second of pump flow by the duty fraction level/255.
                level = max(0.0, min(255.0, d_time))
                return 1000.0 * (level / 255.0)
            if int(msg[33].item()) == 255:
                return max(0.0, d_time) if mode == OD_MODE_BANGBANG else 1000.0
            return 0.0
        except Exception:
            return 0.0

    def _od_bangbang_warmup(self, device, port, od_raw, target_od):
        """Bang-bang control with a timed handoff to clamp/PI.

        Phase 1 — pre-target:  bang-bang while OD is climbing to target.
        Phase 2 — warm-up:     bang-bang for BB_WARMUP_SECONDS after target is first hit.
        Phase 3 — handoff:     switch to clamp mode; PI controller owns OD from here on.

        The hit-time is latched on first reaching target and is NOT reset on
        subsequent dips, so OD oscillation during bang-bang doesn't restart the clock.
        """
        dv = device["DeviceVariables"]

        # Phase 1 — haven't reached target yet: latch the clock once we do.
        if dv["bb_target_hit_time"] is None:
            if od_raw >= target_od:
                dv["bb_target_hit_time"] = time.time()
                print(f"[{port}] OD reached target — bang-bang warm-up started ({BB_WARMUP_SECONDS}s)")
            utils.bangbang(currentOD=od_raw, targetOD=target_od, port=port)
            return

        # Phase 2 — warm-up window still open: keep bang-banging.
        elapsed = time.time() - dv["bb_target_hit_time"]
        if elapsed < BB_WARMUP_SECONDS:
            utils.bangbang(currentOD=od_raw, targetOD=target_od, port=port)
            return

        # Phase 3 — warm-up complete: hand off to clamp and run it on this cycle.
        self._set_od_mode(device, port, OD_MODE_CLAMP)
        dv["bb_target_hit_time"] = None   # cleared in case mode is later switched back to bang-bang
        print(f"[{port}] Bang-bang warm-up complete ({elapsed:.0f}s) → clamp/PI now active")
        device["ML"]["odCont"].update(current_od=od_raw, port=port, target_od=target_od)

    def _init_device_ml(self, port):
        """Attach the per-device filters/controllers (Kalman, OD PI, temp PI, thermal corr)."""
        print(f"[INFO] Initializing filters for port {port}")
        calib_scale = (_Vars.config.get("CalibrationConst", 5.1) / 5.1) ** 2
        model = _Vars.config.get("KalmanModel", "scalar")
        # CV model: Q is acceleration noise (much smaller than the random-walk Q);
        # scalar model: Q is the OD process variance. R is measurement variance for both.
        q_default = 1e-6 if str(model).lower() in ("cv", "2state") else 1e-5
        _Vars.activeDevices[port]["ML"] = {
            "Predictor": None,
            "Kalman":    make_od_filter(model, Q=q_default * calib_scale, R=5e-4 * calib_scale,
                                        x0=0.0, P0=1.0, dt=1.0, window=31, threshold_sigma=5.0,
                                        C=_Vars.config.get("CalibrationConst", 5.1)),
            "odCont":    ODCont(),
            "tempPI":    TempPIController(Kp=_Vars.config.get("TempPI_Kp", 800.0), Ki=_Vars.config.get("TempPI_Ki", 30.0)),
        }
        _Vars.activeDevices[port]["corr"] = ODThermalCorrector(
            tau_s=10, b0=0.04122173, b1=-0.00190831, dt_s=1.0, use_dTdt=False
        )

    def set_kalman_model(self, model, Q=None, R=None):
        """Rebuild every port's OD filter with the chosen model ('scalar' or 'cv').
        Q/R, if given, are the (un-scaled) process/measurement variances from the GUI;
        they're scaled by the per-device calibration like the startup defaults."""
        is_cv = str(model).strip().lower() in ("cv", "2state", "2-state")
        q_def = 1e-6 if is_cv else 1e-5
        for port in list(_Vars.ports):
            dev = _Vars.activeDevices.get(port)
            if dev is None or "ML" not in dev:
                continue
            calib_scale = (_Vars.config.get("CalibrationConst", 5.1) / 5.1) ** 2
            qq = (Q if Q is not None else q_def) * calib_scale
            rr = (R if R is not None else 5e-4) * calib_scale
            dev["ML"]["Kalman"] = make_od_filter("cv" if is_cv else "scalar",
                                                 Q=qq, R=rr, x0=0.0, P0=1.0,
                                                 dt=1.0, window=31, threshold_sigma=5.0,
                                                 C=dev["Coefficients"].get("CalibrationConst", 5.1))
        _Vars.config["KalmanModel"] = "cv" if is_cv else "scalar"
        print(f"[INFO] OD filter model set to {'CV (2-state)' if is_cv else 'scalar'} on all ports.")

    def provision_over_serial(self):
        """Push this host's LAN IP to the MCU over USB serial before connecting.
        Fallback for when the UDP auto-IP broadcast doesn't reach the board.
        Sends 'PROV IP=<ip>\\n'; the firmware stores it in NVS and unicasts telemetry there.

        [Serial] Port = auto writes to EVERY connected serial port — non-ESP32 devices
        ignore it (firmware only acts on a valid PROV line). Or set an explicit device
        path (COM11 / /dev/cu.usbmodemXXXX). Close idf.py monitor first (port must be free)."""
        cfg  = _Vars.config
        com  = (cfg.get("SerialPort", "") or "auto").strip()
        baud = int(cfg.get("SerialBaud", 115200))
        try:
            import serial  # pyserial
            from serial.tools import list_ports
        except ImportError:
            print("[WARN] pyserial not installed (pip install pyserial) — skipping serial provision.")
            return

        if com.lower() == "auto":
            ports = [p.device for p in list_ports.comports()]
            if not ports:
                print("[WARN] Serial provision: no serial ports found.")
                return False
            print(f"[INFO] Serial provision: attempting all {len(ports)} port(s): {ports}")
        else:
            ports = [com]

        ip  = self._get_local_ip()
        msg = f"PROV IP={ip}\n".encode()

        def _attempt(dev, result):
            # Runs in a worker thread so a port that blocks on open (e.g. a phantom
            # COM port) can be abandoned via join-timeout instead of freezing startup.
            try:
                with serial.Serial(dev, baud, timeout=1, write_timeout=2) as s:
                    time.sleep(2.0)            # settle (opening the port may reset the board)
                    for _ in range(5):         # repeat to catch the firmware's read window
                        s.write(msg); s.flush(); time.sleep(0.4)
                result.append(True)
            except Exception as e:
                result.append(e)

        sent_any = False
        for dev in ports:
            result = []
            th = threading.Thread(target=_attempt, args=(dev, result), daemon=True)
            th.start()
            th.join(timeout=6.0)               # bound each port so a dead one can't hang us
            if not result:
                print(f"[WARN] Serial provision: {dev} unresponsive (port busy/phantom) — skipped.")
            elif result[0] is True:
                print(f"[INFO] Serial provision: sent host IP {ip} to {dev}")
                sent_any = True
            else:
                print(f"[WARN] Serial provision: {dev} failed: {result[0]}")
        if not sent_any:
            print("[WARN] Serial provision: no serial port accepted the write.")

    def initStart(self):
        while not _Vars.initState:
            pass
        # Offline/demo mode — no hardware needed, simulate reactors so the GUI runs.
        if _Vars.config.get("Offline", False):
            self.offline_start()
            return
        try:
            # Optional: push host IP over serial before the UDP connect (auto-IP fallback).
            isPort = False
            if _Vars.config.get("SerialProvision", False):
                isPort = self.provision_over_serial()
            self.connect2server(isPort)
            for port in _Vars.ports:
                self._init_device_ml(port)
                # default-argument capture fixes the lambda closure bug —
                # without `p=port`, every thread would share the final loop value of `port`
                threading.Thread(
                    target=lambda p=port: self.main(port=p), daemon=True
                ).start()
        except Exception as e:
            print(f"[ERROR] initStart failed: {e}")
        finally:
            # Always release the GUI, even if zero devices were found or a step threw,
            # so the window opens and the operator can Upload IP / Rescan.
            _Vars.notConnected = True

    def rescan_devices(self):
        """Scan for reactors that weren't connected at startup and bring them online
        live (bind, init filters, start their worker thread). Returns the list of newly
        added ports. Safe to call repeatedly — ports already connected are skipped."""
        if _Vars.config.get("Offline", False):
            return []
        multiplex = _Vars.config.get("Multiplex", True)
        ports_to_scan = range(12340, 12346) if multiplex else [_Vars.config.get("Port", 12340)]
        new_ports = []
        for port in ports_to_scan:
            if port in _Vars.ports:
                continue                      # already connected — leave it alone
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(("0.0.0.0", port))
                sock.settimeout(1.25)
                try:
                    data, addr = sock.recvfrom(64)
                    _Vars.ports.append(port)
                    _Vars.activeDevices[port] = self.init_device_struct(sock, addr)
                    self._init_device_ml(port)
                    threading.Thread(target=lambda p=port: self.main(port=p), daemon=True).start()
                    new_ports.append(port)
                    print(f"[INFO] Rescan: connected device on port {port} ({addr})")
                except socket.timeout:
                    sock.close()
            except OSError as e:
                print(f"[WARN] Rescan: port {port} unavailable: {e}")
                if sock is not None:
                    try: sock.close()
                    except Exception: pass
        print(f"[INFO] Rescan complete — new devices: {new_ports or 'none'}")
        return new_ports

    def offline_start(self):
        """Spin up simulated reactors (no sockets) and feed synthetic growth/temperature
        curves so the whole UI can be exercised without any hardware connected.
        Toggle via [Debug] Offline = True in config.ini.

        Data source: the physics-based digital twin (DigitalTwin.SensorTwinEngine)
        when [Debug] Twin = True, else the lightweight logistic simulator. The twin
        is imported lazily so its absence just falls back to the simple model."""
        n = int(_Vars.config.get("OfflinePorts", 2))
        use_twin = bool(_Vars.config.get("Twin", False))
        if use_twin:
            try:
                import DigitalTwin  # noqa: F401  — probe availability up front
            except Exception as e:
                print(f"[WARN] Digital twin unavailable ({e}); using the simple simulator.")
                use_twin = False
        worker = self.simulate_twin if use_twin else self.simulate
        _Vars.ports.clear()
        for i in range(n):
            port = 12340 + i
            _Vars.ports.append(port)
            _Vars.activeDevices[port] = self.init_device_struct(sock=None, addr=("offline", port))
            self._init_device_ml(port)
            threading.Thread(target=lambda p=port: worker(port=p), daemon=True).start()
        _Vars.notConnected = True
        src = "DIGITAL TWIN" if use_twin else "simple simulator"
        print(f"[INFO] *** OFFLINE MODE ({src}) *** simulating {n} reactor(s): {_Vars.ports}")

    def simulate_twin(self, port):
        """Offline data source backed by the physics-based digital twin: LED /
        optics / thermal plant / biomass growth + detector noise, scatter, and
        bubble disturbances. It supplies realistic raw Main/Ref/Scatter signals +
        temperature; the normal GUI Beta/OD/Kalman pipeline reconstructs OD from
        them exactly as it would for a live reactor. Selected via [Debug] Twin=True.

        TimeScale works the same as the simple simulator: each loop advances the
        twin by one simulated second (engine dt=1s); TimeScale compresses the
        wall-clock spacing between loops."""
        from DigitalTwin import build_engine, od_ceiling
        device = _Vars.activeDevices[port]
        device["_initTime"] = time.time()
        flush_interval = float(_Vars.config.get("FlushIntervalSeconds", 5))
        device["_last_flush"] = time.time()
        time_scale = max(1.0, float(_Vars.config.get("TimeScale", 1.0)))
        real_dt = 1.0 / time_scale
        target_t = device["DeviceVariables"]["ControlValues"]["Temperature"][0]
        # Feed the twin the SAME calibration constant the GUI reconstructs OD with,
        # so the twin's internal od and the GUI's computed OD can't diverge.
        calib_c  = float(device["Coefficients"]["CalibrationConst"] or 5.1)
        alpha    = float(_Vars.config.get("TwinAlphaAtt", 2.5e-9))
        cap      = float(_Vars.config.get("TwinCarryingCap", 1e9))
        engine = build_engine(
            target_temp_C=float(target_t),
            seed=port,                                   # per-reactor trajectory
            mu_mean=_Vars.config.get("TwinMuMean", 0.002),
            inoculation_time_s=_Vars.config.get("TwinInoculationS", 120.0),
            lag_time_s=_Vars.config.get("TwinLagS", 300.0),
            carrying_capacity_cells_ml=cap,
            alpha_att=alpha,
            od_calibration_constant=calib_c,
            bubbles_enabled=_Vars.config.get("TwinBubbles", True),
            bubble_micro_rate=_Vars.config.get("TwinBubbleMicroRate", 0.09),
            bubble_large_rate=_Vars.config.get("TwinBubbleLargeRate", 0.0004),
            bubble_churn_rate=_Vars.config.get("TwinBubbleChurnRate", 1.0 / 15000.0),
        )
        # OD_max is set by optics+biology (C * alpha * cap / ln10), NOT by growth
        # rate — log it so a low ceiling is obvious rather than looking like a bug.
        print(f"[INFO] Digital-twin reactor on port {port} "
              f"(seed={port}, Tset={target_t}°C, TimeScale={time_scale:g}, "
              f"OD ceiling ≈ {od_ceiling(calib_c, alpha, cap):.2f})")
        engine.config.thermal.media_temp_C = float(_Vars.config.get("TwinMediaTemp", 25.0))
        engine.config.bubbles.circulation_threshold_pct = float(
            _Vars.config.get("TwinCircThreshold", 50.0))
        cmp_filters = ODPipelineCompare(C=calib_c)

        # --- turbidostat actuation ------------------------------------------
        # The twin runs the SAME control code as live hardware and a dose really
        # removes cells, so the loop is closed and testable offline.
        twin_turb  = bool(_Vars.config.get("TwinTurbidostat", True))
        pump_flow  = float(_Vars.config.get("TwinPumpFlow", 0.5))       # mL/s
        vessel_mL  = float(_Vars.config.get("TwinVesselVolume", 50.0))
        lag_s      = float(_Vars.config.get("TwinTransportLagS", 4.0))
        pending_doses = []      # (arrival_sim_time, volume_mL) — transport delay
        sim_now = 0.0
        while _Vars.initState:
            # Live coupling: GUI temperature setpoint → twin thermal target;
            # aeration/stirrer → micro-bubble rate, circulation → large-bubble rate.
            engine.set_target_temp(device["DeviceVariables"]["ControlValues"]["Temperature"][0])
            spd = device["DeviceVariables"]["MotorSpeeds"]
            engine.set_motor_speeds(stirrer=spd[1], circulation=spd[0])
            row = engine.step()
            main = row["main_signal"]; ref = row["ref_signal"]
            scat = row["scatter_signal"]; temp = row["temperature_C"]

            for ch in range(3):
                # channel 1 is the pair the GUI's calcOD uses; 0/2 are coherent
                # decoys (same magnitude, tiny per-channel jitter) so the Debug
                # tab's multi-channel view stays sensible.
                j = 1.0 if ch == 1 else (1.0 + random.uniform(-0.003, 0.003))
                mv, rv, sv = main * j, ref * j, scat * j
                device["SensorsRaw"]["Main"][ch].append(mv)
                device["SensorsAvg"]["Main"][ch].append(utils.movingAvg(mv, device["BufferArrs"]["Main"][ch], port))
                device["SensorsRaw"]["Ref"][ch].append(rv)
                device["SensorsAvg"]["Ref"][ch].append(utils.movingAvg(rv, device["BufferArrs"]["Ref"][ch], port))
                device["SensorsRaw"]["Scatter"][ch].append(sv)
                device["SensorsAvg"]["Scatter"][ch].append(utils.movingAvg(sv, device["BufferArrs"]["Scatter"][ch], port))

            device["SensorsRaw"]["Temp"].append(temp)
            device["SensorsAvg"]["Temp"].append(utils.movingAvg(temp, device["BufferArrs"]["Temp"], port))
            device["Time"]["Software"].append(round(row["time_s"], 2))
            device["DeviceVariables"]["RelayState"].append(1 if engine.thermal.heater_on else 0)
            device["DeviceVariables"]["MotorState"].append(self._pump_state(device))
            _cu, _ce, _ci = self._control_state(device)
            device["DeviceVariables"]["ControlU"].append(_cu)
            device["DeviceVariables"]["ControlError"].append(_ce)
            device["DeviceVariables"]["ControlI"].append(_ci)
            device["PacketLoss"][0] += 1
            device["_lastRx"] = time.time()

            try:
                utils.calcBeta(port=port)
                od = utils.calcOD(port=port)
                device["OpticalDensity"]["OD"].append(od)
                device["OpticalDensity"]["KOD"].append(
                    device["ML"]["Kalman"].update(device["SensorsRaw"]["Main"][1][-1],
                                                  device["SensorsRaw"]["Ref"][1][-1],
                                                  utils.betaValue(port),
                                                  scatter=device["SensorsRaw"]["Scatter"][1][-1],
                                                  allow_despike=self._despike_allowed(device)))
            except Exception:
                device["OpticalDensity"]["OD"].append(row["od"])
                device["OpticalDensity"]["KOD"].append(row["od"])

            # Diagnostic: same raw signals through three filtering domains, against
            # the twin's ground-truth OD. Only the twin can supply od_true.
            try:
                fc = cmp_filters.update(main, ref, utils.betaValue(port), scatter=scat)
                device["FilterCompare"]["A"].append(fc["A"])
                device["FilterCompare"]["C"].append(fc["C"])
                device["FilterCompare"]["D"].append(fc["D"])
                device["FilterCompare"]["E"].append(fc["E"])
                device["FilterCompare"]["True"].append(row["od_true"])
            except Exception as e:
                print(f"[WARN] filter-compare on port {port}: {e}")

            # ---- closed-loop turbidostat -----------------------------------
            if twin_turb and device["DeviceVariables"]["Turbidostat"]:
                try:
                    od_now = device["OpticalDensity"]["OD"][-1]
                    kod_now = device["OpticalDensity"]["KOD"][-1]
                    od_ctrl = od_now
                    if (_Vars.config.get("ControlUsesFilteredOD", True)
                            and len(device["OpticalDensity"]["OD"]) > 30):
                        od_ctrl = kod_now
                    target_od = device["DeviceVariables"]["ControlValues"]["OD"][0]
                    # FORCED: OD control is locked to clamp/ODCont (proportional
                    # speed, 0-255 on byte 24). Bang-bang / hysteresis are disabled.
                    if int(device["OutgoingMessage"][32].item()) != OD_MODE_CLAMP:
                        self._set_od_mode(device, port, OD_MODE_CLAMP)
                    device["ML"]["odCont"].update(current_od=od_ctrl, port=port,
                                                  target_od=target_od)

                    # Command -> volume -> SCHEDULED for arrival at the window.
                    dose_ms = self._commanded_dose_ms(device)
                    if dose_ms > 0:
                        vol = pump_flow * (dose_ms / 1000.0)
                        pending_doses.append((sim_now + lag_s, vol))

                    # Fresh media pumped `lag_s` ago now reaches the sensor: the
                    # cells it displaced disappear from the optical path only NOW.
                    # This dead time is what a turbidostat overshoots through.
                    while pending_doses and pending_doses[0][0] <= sim_now:
                        _, vol = pending_doses.pop(0)
                        # Does BOTH: displaces cells AND cools the vessel, because
                        # fresh media arrives at ambient, not at setpoint.
                        engine.add_fresh_media(vol, vessel_mL)
                except Exception as e:
                    print(f"[WARN] twin turbidostat on port {port}: {e}")
            sim_now += 1.0

            now = time.time()
            if now - device["_last_flush"] >= flush_interval:
                device["_last_flush"] = now
                submitFlush(port)
            if real_dt > 0:
                time.sleep(real_dt)

    def simulate(self, port):
        """Generate plausible sensor data once per second for one fake reactor.
        Reuses the real Beta/OD/Kalman pipeline so plots and tiles behave like a live run.

        Time compression: `[Debug] TimeScale` (default 1.0) is how many *simulated*
        seconds elapse per real second. TimeScale=1000 means each iteration still
        represents 1 simulated second, but they fire ~1000x faster in wall-clock,
        so a 6-hour rolling window fills up in ~22 real seconds. All in-loop time
        reasoning (heater dwell, stored Time.Software) uses simulated seconds so
        the plots and controls behave as if it were real-time — only the flush
        cadence stays on wall-clock (that's disk I/O, not simulation)."""
        device = _Vars.activeDevices[port]
        device["_initTime"] = time.time()
        flush_interval = float(_Vars.config.get("FlushIntervalSeconds", 5))
        device["_last_flush"] = time.time()
        time_scale = max(1.0, float(_Vars.config.get("TimeScale", 1.0)))
        sim_dt = 1.0                          # each iteration = 1 simulated second
        real_dt = sim_dt / time_scale         # ...packed into this many real seconds
        if time_scale != 1.0:
            print(f"[INFO] Offline TimeScale={time_scale:g} — 1 real second = {time_scale:g} simulated seconds on port {port}")
        C   = device["Coefficients"]["CalibrationConst"] or 1.0
        cap = 4.0 + random.uniform(-1.5, 0.5)   # carrying capacity varies per reactor
        k   = 0.05
        t0  = 40.0
        temp = 25.0
        heater_on = False
        sim_now = 0.0                          # simulated seconds since start
        last_switch_sim = 0.0
        n = 0
        while _Vars.initState:
            t = round(sim_now, 2)
            # Heater model — honours the GUI heater mode + bang-bang delay so the
            # temperature controls are observable offline. (Real hardware does this
            # in firmware; here we mirror the logic so the demo reacts to settings.)
            tctrl       = device["DeviceVariables"]["ControlValues"]["Temperature"]
            target_t    = tctrl[0]
            band        = tctrl[1]
            delay_s     = max(tctrl[2] / 1000.0, 0.001)   # bang-bang min dwell (sim seconds)
            heater_mode = int(device["OutgoingMessage"][18].item())
            if heater_mode == 1:                          # hysteresis
                if heater_on and temp >= target_t + band:
                    heater_on = False
                elif (not heater_on) and temp <= target_t - band:
                    heater_on = True
            else:                                         # bang-bang (0) / PI (2): dwell-limited
                if (sim_now - last_switch_sim) >= delay_s:
                    new_on = temp < target_t
                    if new_on != heater_on:
                        heater_on = new_on
                        last_switch_sim = sim_now
            temp += (0.30 if heater_on else -0.18) + random.uniform(-0.04, 0.04)

            od_true = cap / (1.0 + math.exp(-k * (n - t0)))
            R = 1000.0 + random.uniform(-3, 3)
            M = 1000.0 / (10 ** (od_true / C)) + random.uniform(-1, 1)
            if M <= 0:
                M = 0.01

            for ch in range(3):
                mv = M if ch == 1 else 1000 + random.uniform(-5, 5)
                rv = R if ch == 1 else 1000 + random.uniform(-5, 5)
                sv = 200 + random.uniform(-5, 5)
                device["SensorsRaw"]["Main"][ch].append(mv)
                device["SensorsAvg"]["Main"][ch].append(utils.movingAvg(mv, device["BufferArrs"]["Main"][ch], port))
                device["SensorsRaw"]["Ref"][ch].append(rv)
                device["SensorsAvg"]["Ref"][ch].append(utils.movingAvg(rv, device["BufferArrs"]["Ref"][ch], port))
                device["SensorsRaw"]["Scatter"][ch].append(sv)
                device["SensorsAvg"]["Scatter"][ch].append(utils.movingAvg(sv, device["BufferArrs"]["Scatter"][ch], port))

            device["SensorsRaw"]["Temp"].append(temp)
            device["SensorsAvg"]["Temp"].append(utils.movingAvg(temp, device["BufferArrs"]["Temp"], port))
            device["Time"]["Software"].append(t)
            device["DeviceVariables"]["RelayState"].append(1 if heater_on else 0)
            device["DeviceVariables"]["MotorState"].append(self._pump_state(device))
            _cu, _ce, _ci = self._control_state(device)
            device["DeviceVariables"]["ControlU"].append(_cu)
            device["DeviceVariables"]["ControlError"].append(_ce)
            device["DeviceVariables"]["ControlI"].append(_ci)
            device["PacketLoss"][0] += 1
            device["_lastRx"] = time.time()   # liveness for the GUI status dot

            try:
                utils.calcBeta(port=port)
                od = utils.calcOD(port=port)
                device["OpticalDensity"]["OD"].append(od)
                device["OpticalDensity"]["KOD"].append(
                    device["ML"]["Kalman"].update(device["SensorsAvg"]["Main"][1][-1],
                                                  device["SensorsAvg"]["Ref"][1][-1],
                                                  utils.betaValue(port)))
            except Exception:
                device["OpticalDensity"]["OD"].append(od_true)
                device["OpticalDensity"]["KOD"].append(od_true)

            # Same streaming flush + rolling trim as the live comms loop, handed
            # off to the background flusher so a slow disk doesn't stall the
            # simulator (matters at high TimeScale where samples arrive fast).
            now = time.time()
            if now - device["_last_flush"] >= flush_interval:
                device["_last_flush"] = now
                submitFlush(port)
            n += 1
            sim_now += sim_dt
            if real_dt > 0:
                time.sleep(real_dt)

    def connect2server(self, isPort):
        if isPort:
            time.sleep(8)
        _Vars.ports.clear()
        multiplex = _Vars.config.get("Multiplex", True)
        ports_to_scan = range(12340, 12346) if multiplex else [_Vars.config.get("Port", 12340)]
        print(f"[INFO] {'Scanning ports ' + str(list(ports_to_scan)) if multiplex else 'Connecting to port ' + str(ports_to_scan[0])}...")
        for port in ports_to_scan:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(("0.0.0.0", port))
                sock.settimeout(1.25)
                print(f"[INFO] Bound to port {port}. Waiting for data...")
                try:
                    data, addr = sock.recvfrom(64)
                    print(f"[OK] Data from {addr} on port {port}: {data}")
                    _Vars.ports.append(port)
                    _Vars.activeDevices[port] = self.init_device_struct(sock, addr)
                except socket.timeout:
                    print(f"[WARN] No data on port {port}. Closing.")
                    sock.close()
            except OSError as e:
                print(f"[WARN] Port {port} unavailable: {e}")
                sock.close()
        _Vars.notConnected = True

    @staticmethod
    def _build_outgoing_packet(cfg) -> np.ndarray:
        import struct

        def pack_f(val, msg, offset):
            b = struct.pack("f", float(val))
            for i in range(4):
                msg[offset + i] = b[i]

        temp  = list(cfg.get("Temperature", [37.0, 0.3, 5.0, 2.0]))
        od    = list(cfg.get("OD",          [0.20, 5.0, 0.3, 0.0]))
        spds  = list(cfg.get("MotorSpeeds",     [0, 0, 0, 0]))
        dirs  = list(cfg.get("MotorDirections", [0, 0, 0, 0]))

        heater_mode_map = {"bangbang": 0, "hyst": 1, "pi": 2}
        od_mode_map     = {"bangbang": 0, "hyst": 1, "clampod": 2}

        msg = np.zeros((60, 1), dtype=np.uint8)

        # Heater block (bytes 2–18)
        pack_f(temp[0], msg, 2)   # temp_target
        pack_f(temp[1], msg, 6)   # temp_bandwidth
        pack_f(cfg.get("TemperatureOffset", 0.0), msg, 10)  # temp_offset ([Offsets] TemperatureOffset)
        pack_f(temp[2], msg, 14)  # heater_delay_ms
        msg[18] = heater_mode_map.get(cfg.get("HeaterMode", "hyst"), 1)

        # OD block (bytes 20–33)
        od_mode = od_mode_map.get(cfg.get("TurbidostatMode", "clampod"), 2)
        pack_f(od[0], msg, 20)    # OD_target
        # byte 24: in clamp mode (2) this is the 0-255 pump LEVEL and must start at
        # 0 (holding) — seeding od[1] (the bang-bang dose delay, e.g. 10000) made
        # the Motor plot spike to 10000 at startup until ODCont's first update.
        # In bang-bang/hyst it keeps its original meaning (OD_bb_delay / d_time).
        pack_f(0.0 if od_mode == 2 else od[1], msg, 24)    # OD_bb_delay / d_time / clamp level
        pack_f(od[3], msg, 28)    # OD_hyst_bandwidth
        msg[32] = od_mode
        msg[34] = 1 if cfg.get("Turbidostat", False) else 0  # gates MCU pump ownership
        # byte 35 = pinch valve pair; MCU drives high-side CH2/CH3 complementarily
        # (0 = CH2 on / CH3 off = rest, 1 = CH2 off / CH3 on). GUI Pinch State button.
        msg[35] = 0
        # byte 0 = LED_mode (0 normal / 1 modulated dark-subtract), byte 52 = N-averaging
        msg[0]  = 0
        msg[52] = 1
        # byte 53 = auxiliary CH0 motor speed (0-255 digipot level), always
        # driven. Stepper channel PUMP_CH0, not a magfan -> not a percent.
        try:    msg[53] = int(cfg.get("CH0Speed", 0)) & 0xFF
        except Exception: msg[53] = 0

        # Motor block (bytes 36–42)
        msg[36] = int(dirs[0])    # cycle dir
        msg[37] = int(spds[0])    # cycle spd
        msg[38] = int(spds[1])    # fan spd
        msg[39] = int(dirs[2])    # waste dir
        msg[40] = int(spds[2])    # waste spd
        msg[41] = int(dirs[3])    # fresh dir
        msg[42] = int(spds[3])    # fresh spd

        # Auto host-IP discovery (bytes 43-46 = octets, 47 = valid flag).
        # The MCU reads these to learn where to send telemetry, so no manual
        # IP entry is needed. This travels over UDP, NOT the serial line, so it
        # is gated on [Network] AutoIP — independent of [Serial] Provision. (It
        # used to read SerialProvision, which meant turning serial off also
        # silently disabled the Wi-Fi auto-IP path.)
        if _Vars.config.get("AutoIP", True):
            octets = communications._get_local_ip().split(".")
            if len(octets) == 4:
                msg[43], msg[44], msg[45], msg[46] = (int(o) for o in octets)
                msg[47] = 1

        return msg

    @staticmethod
    def _get_local_ip() -> str:
        """Best-effort LAN IP of this host (the egress interface toward the line).
        No packets are actually sent — connect() just selects the route."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            s.close()

    def init_device_struct(self, sock, addr):
        cfg = _Vars.config
        # High-volume per-sample columns → preallocated float32 ring buffers.
        # Everything sized to WindowSeconds so the DRAM footprint is bounded
        # up-front and appends never allocate. Small / rarely-appended lists
        # (Beta, EstimatedDt, BufferArrs, PacketLoss) stay Python lists — the
        # per-item overhead there is negligible and the code that touches them
        # already treats them as growable lists.
        cap = int(cfg.get("WindowSeconds", 21600))
        rb = lambda: RingBuffer(cap)
        return {
            "Network": {
                "Socket": sock,
                "Addr":   addr,
            },
            "SensorsRaw": {
                "Main":    [rb(), rb(), rb()],
                "Ref":     [rb(), rb(), rb()],
                "Scatter": [rb(), rb(), rb()],
                "Temp":    rb(),
            },
            "BufferArrs": {
                "BufferLength": 10,
                "Main":    [[], [], []],
                "Ref":     [[], [], []],
                "Scatter": [[], [], []],
                "Temp":    [],
            },
            "SensorsAvg": {
                "Main":    [rb(), rb(), rb()],
                "Ref":     [rb(), rb(), rb()],
                "Scatter": [rb(), rb(), rb()],
                "Temp":    rb(),
            },
            "OpticalDensity": {"OD": rb(), "KOD": rb(), "EstimatedDt": []},
            "MLPredictions":  {"RefPred": rb(), "OD": rb()},
            "PacketLoss": [0, 0],
            "sawtoothing": {
                "running":        False,
                "stop_time":      0.0,
                "cooldown_until": 0.0,
            },
            "DeviceVariables": {
                "RelayState":  rb(),
                "MotorState":  rb(),
                "ControlU":    rb(),   # ODCont output u (0-255); logged per sample
                "ControlError": rb(),  # ODCont error e = target_od - current_od
                "ControlI":    rb(),   # ODCont integral term I; logged per sample

                "MotorSpeeds":     list(cfg.get("MotorSpeeds",     [0, 0, 0, 0])),
                "MotorDirections": list(cfg.get("MotorDirections", [0, 0, 0, 0])),
                "CH0Speed":        int(cfg.get("CH0Speed", 0)),
                "ControlValues": {
                    "Temperature": list(cfg.get("Temperature", [37.0, 0.3, 5.0, 2.0])),
                    "OD":          list(cfg.get("OD",          [0.20, 5.0, 0.3, 0.0])),
                },
                "Turbidostat":     cfg.get("Turbidostat",     True),
                "TurbidostatMode": cfg.get("TurbidostatMode", "clampod"),
                "HeaterMode":      cfg.get("HeaterMode",      "hyst"),
                "TODs": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.01, 12],
                "SendMessage": True,
                "bb_target_hit_time": None,  # wall-clock time OD first reached target in bang-bang mode
            },
            "Coefficients": {
                "Beta":              [],           # live blank-window accumulation
                "BetaFinal":         0.0,          # AUTO beta, latched after BetaTime (an output, always computed)
                "BetaPredefined":    float(cfg.get("Beta", 0.0) or 0.0),   # from a prior calibration
                "BetaSource":        cfg.get("BetaSource", "auto"),        # "auto" | "predefined"
                "CalibrationConst":  cfg.get("CalibrationConst",  5.1),
                "TemperatureOffset": cfg.get("TemperatureOffset", 0.0),
            },
            "Time": {"Software": rb(), "BetaTime": cfg.get("BetaTime", 1200), "Hardware": rb()},
            "OutgoingMessage": self._build_outgoing_packet(cfg),
            "corrected_OD": rb(),
            # Simulator-only diagnostic: three OD pipelines run side by side plus
            # the twin's noise-free ground truth. Populated by simulate_twin,
            # plotted on the Debug tab; stays empty for real hardware.
            "FilterCompare": {"A": rb(), "C": rb(), "D": rb(), "E": rb(), "True": rb()},
            # Per-reactor strain label ([Run] Strain default; editable live via the
            # Run menu). Feeds the run filename + the XML sidecar's <Strain>.
            "Strain": str(cfg.get("Strain", "SB5")),
        }

    def listenPorts(self, port):
        try:
            device = _Vars.activeDevices[port]
            sock   = device["Network"]["Socket"]
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                return
            except OSError as e:
                print(f"[ERROR] Socket error on port {port}: {e}")
                return

            device["Network"]["Addr"] = addr
            if not utils.rx_packet_is_valid(data):
                device["PacketLoss"][1] += 1
                return
            device["PacketLoss"][0] += 1
            device["_lastRx"] = time.time()   # for the GUI liveness/staleness dot
            # Unpack — first 60 bytes are 15 little-endian float32s; last 4 are the CRC.
            # One numpy C call vs 15 Python-level struct.unpack calls.
            unpacked = np.frombuffer(data[:60], dtype="<f4").tolist()
            # s0 = main, s1 = scatter, s2 = ref  (indices 0-2, 3-5, 6-8)
            for i in range(len(device["SensorsRaw"]["Main"])):
                device["SensorsRaw"]["Main"][i].append(unpacked[i])
                device["SensorsAvg"]["Main"][i].append(
                    utils.movingAvg(value=unpacked[i],
                                    bufferLoc=device["BufferArrs"]["Main"][i], port=port))

                device["SensorsRaw"]["Scatter"][i].append(unpacked[i + 3])
                device["SensorsAvg"]["Scatter"][i].append(
                    utils.movingAvg(value=unpacked[i + 3],
                                    bufferLoc=device["BufferArrs"]["Scatter"][i], port=port))

                device["SensorsRaw"]["Ref"][i].append(unpacked[i + 6])
                device["SensorsAvg"]["Ref"][i].append(
                    utils.movingAvg(value=unpacked[i + 6],
                                    bufferLoc=device["BufferArrs"]["Ref"][i], port=port))

            device["SensorsRaw"]["Temp"].append(unpacked[9])
            device["SensorsAvg"]["Temp"].append(
                utils.movingAvg(value=unpacked[9],
                                bufferLoc=device["BufferArrs"]["Temp"], port=port))

            device["Time"]["Software"].append(
                round(abs(time.time() - device["_initTime"]), 2))
            device["DeviceVariables"]["RelayState"].append(data[41])

            utils.calcBeta(port=port)

            # OD pipeline: raw → despiked → Kalman
            od_raw = utils.calcOD(port=port)
            device["OpticalDensity"]["OD"].append(od_raw)
            kod = device["ML"]["Kalman"].update(
                device["SensorsRaw"]["Main"][1][-1], device["SensorsRaw"]["Ref"][1][-1],
                utils.betaValue(port),
                scatter=device["SensorsRaw"]["Scatter"][1][-1],
                allow_despike=self._despike_allowed(device))
            device["OpticalDensity"]["KOD"].append(kod)

            # Which OD drives the PUMPS. Historically this was od_raw — unfiltered
            # and un-despiked — so a bubble fed straight into a dilution decision.
            # Bubbles raise main, so OD reads LOW and the controller withholds
            # dilution during churn: a biased controller, not just a noisy one.
            # Measured against twin ground truth, raw made 11.1% of dilution calls
            # wrong (85% during bubbles) vs 2.5% (28%) for the filtered estimate.
            # Set [Kalman] ControlUsesFilteredOD = False to restore the old behaviour.
            od_ctrl = od_raw
            if _Vars.config.get("ControlUsesFilteredOD", True) and len(device["OpticalDensity"]["OD"]) > 30:
                od_ctrl = kod          # filter has settled past its warm-up

            # Ship the control OD to the MCU (bytes 48-51) for its bang-bang test,
            # so host and MCU agree on what "current OD" means.
            utils.packFloatingBytes(utils.convertFloating2Bytes(od_ctrl), idx=48, port=port)
            device["DeviceVariables"]["SendMessage"] = True
            
            if device["ML"]["Kalman"].last_spike:
                print(f"[WARN] Spike detected on port {port} — raw OD: {od_raw:.4f}")
            od_corrected, *_ = device["corr"].update(OD=od_raw, T=unpacked[9])
            device["corrected_OD"].append(od_corrected)
            print(f"[{port}] OD: {od_raw:.4f} | KOD: {device['OpticalDensity']['KOD'][-1]:.4f} | Main: {device['SensorsRaw']['Main'][1][-1]} | Ref: {device['SensorsRaw']['Ref'][1][-1]} | Scatter: {device['SensorsRaw']['Scatter'][1][-1]}"
                  f"| Loss: {utils.packet_loss(device['PacketLoss'][0], device['PacketLoss'][1]):.1f}%")
            
            if device["DeviceVariables"]["Turbidostat"]:
                target_od = device["DeviceVariables"]["ControlValues"]["OD"][0]
                # FORCED: OD control is locked to clamp/ODCont (proportional speed,
                # 0-255 on byte 24). Bang-bang / hysteresis are disabled.
                if int(device["OutgoingMessage"][32].item()) != OD_MODE_CLAMP:
                    self._set_od_mode(device, port, OD_MODE_CLAMP)
                device["ML"]["odCont"].update(current_od=od_ctrl, port=port, target_od=target_od)

            if device["DeviceVariables"]["HeaterMode"] == "pi" and device["SensorsAvg"]["Temp"]:
                device["ML"]["tempPI"].update(
                    current_temp=device["SensorsAvg"]["Temp"][-1],
                    port=port,
                    target_temp=device["DeviceVariables"]["ControlValues"]["Temperature"][0],
                )

            # MotorState: exactly one value per sample reflecting pump ON(1)/OFF(0)
            # across all paths (bang-bang, hysteresis, clamp, or manual override).
            device["DeviceVariables"]["MotorState"].append(self._pump_state(device))
            # ODCont u (dose ms) + error, one per sample, in lockstep with MotorState.
            _cu, _ce, _ci = self._control_state(device)
            device["DeviceVariables"]["ControlU"].append(_cu)
            device["DeviceVariables"]["ControlError"].append(_ce)
            device["DeviceVariables"]["ControlI"].append(_ci)

        except Exception as e:
            # Log the full traceback, not just the message — this handler previously
            # hid a real bug (numpy scalar conversion) for days behind a one-line print.
            print(f"[ERROR] listenPorts({port}): {e.__class__.__name__}: {e}")
            traceback.print_exc()

    def sender(self, port):
        out    = bytes(_Vars.activeDevices[port]["OutgoingMessage"])
        packet = utils.tx_build_packet(out)
        _Vars.activeDevices[port]["Network"]["Socket"].sendto(packet, _Vars.activeDevices[port]["Network"]["Addr"])
        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = False

    def main(self, port):
        # Store the start time on the device dict so each thread has its own —
        # previously self.initTime was shared across all threads, causing races
        _Vars.activeDevices[port]["_initTime"] = time.time()
        device = _Vars.activeDevices[port]
        flush_interval = float(_Vars.config.get("FlushIntervalSeconds", 5))
        device["_last_flush"] = time.time()
        while _Vars.initState:
            self.listenPorts(port)
            if device["DeviceVariables"]["SendMessage"]:
                self.sender(port)
            # Periodic streaming flush + rolling-window trim. Handed off to a
            # background thread via submitFlush so a slow disk (append via
            # pandas.to_csv can hitch tens of ms) never stalls the UDP loop.
            # De-dup in submitFlush prevents a backlog from building up.
            now = time.time()
            if now - device["_last_flush"] >= flush_interval:
                device["_last_flush"] = now
                submitFlush(port)