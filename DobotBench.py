"""DobotBench.py — a minimal, robust test bench for the Dobot Magician.

A clean restart that leaves the full Dobot.py untouched. Scope is deliberately
tiny — three things only:
  1. Connect / Disconnect
  2. Set Home to explicit X/Y/Z coordinates (typed, or captured from the arm)
  3. A Home button that moves to those coordinates

Robustness baked in from what bit us building Dobot.py:
  * pydobot's `wait=True` crashes when a serial read returns None — so we always
    send with wait=False and wait on MOTION SETTLING (pose stops changing).
  * pose() tolerates a bad read and returns the last good value.
  * every Qt button handler is wrapped in guard() — an unhandled exception in a
    PyQt6 slot aborts the whole process.
  * long moves run on a worker thread so the window never freezes; a crash-safe
    line-buffered log captures everything (see __main__).

Run it standalone:  py -3 -u DobotBench.py
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Tuple

DEFAULT_PORT = "COM3"


# ==========================================================================
# Arm: a thin, resilient wrapper over pydobot with a simulation fallback.
# ==========================================================================
class Arm:
    def __init__(self):
        self.dev = None
        self.connected = False
        self.sim = False
        self.port: Optional[str] = None
        self._last: Tuple[float, float, float, float] = (200.0, 0.0, 50.0, 0.0)
        self.home_xyz: Tuple[float, float, float, float] = (200.0, 0.0, 50.0, 0.0)

    # ---- ports -----------------------------------------------------------
    @staticmethod
    def list_ports():
        try:
            from serial.tools import list_ports
            return [p.device for p in list_ports.comports()]
        except Exception:
            return []

    # ---- connection ------------------------------------------------------
    def connect(self, port: Optional[str] = None) -> bool:
        """Open the arm. Falls back to SIMULATION if pydobot/hardware absent.
        Returns True if a real arm connected."""
        if port is None:
            ports = self.list_ports()
            port = DEFAULT_PORT if DEFAULT_PORT in ports else (ports[0] if ports else DEFAULT_PORT)
        try:
            from pydobot import Dobot
        except Exception as e:
            print(f"[bench] pydobot not installed ({e}) — SIMULATION")
            return self._enter_sim(port)
        try:
            self.dev = Dobot(port=port, verbose=False)
            self.connected, self.sim, self.port = True, False, port
            print(f"[bench] connected on {port}")
            return True
        except Exception as e:
            print(f"[bench] connect failed on {port} ({e}) — SIMULATION")
            return self._enter_sim(port)

    def _enter_sim(self, port) -> bool:
        self.dev, self.connected, self.sim, self.port = None, True, True, port
        return False

    def disconnect(self):
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:
                pass
        self.dev, self.connected = None, False
        print("[bench] disconnected")

    # ---- state -----------------------------------------------------------
    def pose(self) -> Tuple[float, float, float, float]:
        """(x, y, z, r). Tolerates pydobot's occasional None read by returning
        the last good pose instead of crashing."""
        if self.sim or self.dev is None:
            return self._last
        try:
            x, y, z, r, *_ = self.dev.pose()
            self._last = (x, y, z, r)
        except (AttributeError, TypeError, IndexError, OSError):
            pass
        return self._last

    def _wait_settle(self, timeout: float = 30.0, eps: float = 0.4, need: int = 3):
        """Block until the arm stops moving (pose stable) — robust against the
        flaky queued-index ack that made moves blur / crash."""
        time.sleep(0.2)                         # let the commanded motion begin
        last = None
        stable = 0
        t0 = time.time()
        while time.time() - t0 < timeout:
            x, y, z, r = self.pose()
            if last is not None and abs(x - last[0]) < eps and abs(y - last[1]) < eps \
                    and abs(z - last[2]) < eps:
                stable += 1
                if stable >= need:
                    return
            else:
                stable = 0
            last = (x, y, z, r)
            time.sleep(0.09)

    # ---- motion ----------------------------------------------------------
    def move_to(self, x: float, y: float, z: float, r: float = 0.0, wait: bool = True):
        if self.sim or self.dev is None:
            self._last = (x, y, z, r)
            if wait:
                time.sleep(0.05)
            return
        from pydobot.enums import PTPMode
        # wait=False never touches the (possibly None) ack; we wait ourselves.
        self.dev._set_ptp_cmd(x, y, z, r, mode=PTPMode.MOVJ_XYZ, wait=False)
        if wait:
            self._wait_settle()

    # ---- home ------------------------------------------------------------
    def set_home(self, x: float, y: float, z: float, r: float = 0.0):
        self.home_xyz = (x, y, z, r)
        print(f"[bench] home set to {self.home_xyz}")

    def home(self, wait: bool = True):
        x, y, z, r = self.home_xyz
        self.move_to(x, y, z, r, wait=wait)
        print(f"[bench] homed to {self.home_xyz}")


# ==========================================================================
# PlateMap: well (row,col) -> robot XY via a least-squares similarity fit.
# ==========================================================================
ROWS, COLS = 4, 6
ROW_LABELS = "ABCD"
DEFAULT_PITCH = 19.3          # ANSI/SLAS 24-well pitch (mm)


def well_name(r: int, c: int) -> str:
    return f"{ROW_LABELS[r]}{c + 1}"


def all_wells():
    return [well_name(r, c) for r in range(ROWS) for c in range(COLS)]


def parse_well(name: str) -> Tuple[int, int]:
    name = name.strip().upper()
    return ROW_LABELS.index(name[0]), int(name[1:]) - 1


class PlateMap:
    """Maps a well name to a robot XY coordinate.

    Fit model = SIMILARITY (rotation + uniform scale + translation), least-
    squares (Umeyama) of the taught corners against the known nominal grid.
    Only 4 DOF, so 4 taught corners over-determine it — teaching noise averages
    out and can't turn into fake skew, and the known 19.3 mm pitch is used
    rather than re-measured. `res_max/res_rms` are the accuracy gauge.
    """

    def __init__(self, pitch: float = DEFAULT_PITCH):
        self.pitch = pitch
        self.taught = {}                       # 'A1' -> (x, y) robot coords
        self.R = None
        self.s = 1.0
        self.t = None
        self.res_rms = 0.0
        self.res_max = 0.0
        self.fitted = False

    def teach(self, name: str, xy: Tuple[float, float]):
        parse_well(name)                        # validate A1..D6
        self.taught[name] = (float(xy[0]), float(xy[1]))
        self.fitted = False

    def n(self) -> int:
        return len(self.taught)

    def _nominal(self, r: int, c: int):
        return (c * self.pitch, r * self.pitch)   # ideal grid, plate frame

    def fit(self):
        import numpy as np
        if len(self.taught) < 2:
            raise RuntimeError("Teach at least 2 corners (ideally all 4).")
        P, Q = [], []
        for name, (x, y) in self.taught.items():
            r, c = parse_well(name)
            P.append(self._nominal(r, c)); Q.append((x, y))
        P = np.array(P, float); Q = np.array(Q, float)
        muP, muQ = P.mean(0), Q.mean(0)
        Pc, Qc = P - muP, Q - muQ
        H = (Qc.T @ Pc) / len(P)
        U, D, Vt = np.linalg.svd(H)
        # ALLOW reflection: a rigid plate maps to either handedness in the robot
        # frame depending on how it's oriented vs the arm's Y axis. Forcing a
        # proper rotation here mis-fit a mirror-placed plate by ~50 mm. R=U@Vt is
        # the best-fit orthogonal transform (rotation OR reflection).
        R = U @ Vt
        varP = (Pc ** 2).sum() / len(P)
        s = float(D.sum() / varP) if varP > 0 else 1.0
        t = muQ - s * (R @ muP)
        self.R, self.s, self.t = R, s, t
        errs = []
        for (px, py), (qx, qy) in zip(P, Q):
            pr = s * (R @ np.array([px, py])) + t
            errs.append(((pr[0] - qx) ** 2 + (pr[1] - qy) ** 2) ** 0.5)
        errs = np.array(errs)
        self.res_rms = float((errs ** 2).mean() ** 0.5)
        self.res_max = float(errs.max())
        self.fitted = True

    def well_xy(self, name: str) -> Tuple[float, float]:
        import numpy as np
        if not self.fitted:
            self.fit()
        r, c = parse_well(name)
        q = self.s * (self.R @ np.array(self._nominal(r, c), float)) + self.t
        return float(q[0]), float(q[1])

    def report(self) -> str:
        if self.n() == 0:
            return "no corners taught"
        try:
            self.fit()
        except Exception as e:
            return str(e)
        return (f"fit over {self.n()} pts — residual rms {self.res_rms:.2f} mm, "
                f"max {self.res_max:.2f} mm  (scale {self.s:.4f})")


# ==========================================================================
# UI
# ==========================================================================
def build_window():
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QLineEdit, QComboBox,
    )
    from PyQt6.QtCore import QTimer

    ACCENT, MUTED, TEXT, DANGER = "#1bb89a", "#8a93a3", "#e6e9ef", "#e05a5a"
    arm = Arm()
    plate = PlateMap()

    win = QMainWindow()
    win.setWindowTitle("Dobot Bench")
    central = QWidget()
    root = QVBoxLayout(central)

    status = QLabel("Disconnected")
    status.setStyleSheet(f"color:{MUTED}; font-weight:600;")
    pose_lbl = QLabel("X — Y — Z — R —")
    pose_lbl.setStyleSheet(f"color:{TEXT}; font-family:Consolas; font-size:15px;")

    hx, hy, hz = QLineEdit("200"), QLineEdit("0"), QLineEdit("50")
    for f in (hx, hy, hz):
        f.setFixedWidth(70)

    busy = {"v": False}                          # true while a threaded move runs

    def refresh():
        # Never raise from here (QTimer slot): swallow and show.
        try:
            if not arm.connected:
                return
            status.setText("SIMULATION" if arm.sim else f"Connected · {arm.port}")
            status.setStyleSheet(f"color:{ACCENT}; font-weight:600;")
            x, y, z, r = arm.pose()
            pose_lbl.setText(f"X {x:7.1f}   Y {y:7.1f}   Z {z:7.1f}   R {r:6.1f}")
        except Exception as e:
            status.setText(f"read error: {e}")
            status.setStyleSheet(f"color:{DANGER}; font-weight:600;")

    def guard(fn):
        """Keep a raised handler exception from aborting the PyQt6 process."""
        def slot(*_):
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc()
                status.setText(f"⚠ {e}")
                status.setStyleSheet(f"color:{DANGER}; font-weight:600;")
        return slot

    def run_bg(fn, msg):
        """Run a blocking arm op on a worker thread so the UI stays alive."""
        if busy["v"] or not arm.connected:
            return
        busy["v"] = True
        status.setText(msg)
        status.setStyleSheet(f"color:{ACCENT}; font-weight:600;")

        def worker():
            try:
                fn()
            except Exception as e:
                print(f"[bench] op error: {e}")
            finally:
                busy["v"] = False
        threading.Thread(target=worker, daemon=True).start()

    # --- handlers ---
    def on_connect():
        arm.connect()
        refresh()

    def on_disconnect():
        arm.disconnect()
        status.setText("Disconnected")
        status.setStyleSheet(f"color:{MUTED}; font-weight:600;")
        pose_lbl.setText("X — Y — Z — R —")

    def on_set_home():
        arm.set_home(float(hx.text() or 0), float(hy.text() or 0), float(hz.text() or 0))
        status.setText(f"home set: ({hx.text()}, {hy.text()}, {hz.text()})")

    def on_capture():
        x, y, z, r = arm.pose()
        hx.setText(f"{x:.1f}"); hy.setText(f"{y:.1f}"); hz.setText(f"{z:.1f}")
        arm.set_home(x, y, z, r)
        status.setText(f"home captured: ({x:.1f}, {y:.1f}, {z:.1f})")

    def on_home():
        run_bg(lambda: arm.home(wait=True), "homing → coordinates…")

    # --- layout: connection row ---
    row1 = QHBoxLayout()
    b_conn = QPushButton("Connect"); b_conn.clicked.connect(guard(on_connect))
    b_disc = QPushButton("Disconnect"); b_disc.clicked.connect(guard(on_disconnect))
    row1.addWidget(b_conn); row1.addWidget(b_disc); row1.addWidget(status); row1.addStretch()
    root.addLayout(row1)
    root.addWidget(pose_lbl)

    # --- layout: home coordinates row ---
    row2 = QHBoxLayout()
    row2.addWidget(QLabel("Home  X:")); row2.addWidget(hx)
    row2.addWidget(QLabel("Y:")); row2.addWidget(hy)
    row2.addWidget(QLabel("Z:")); row2.addWidget(hz)
    b_set = QPushButton("Set Home"); b_set.clicked.connect(guard(on_set_home))
    b_cap = QPushButton("Capture current"); b_cap.clicked.connect(guard(on_capture))
    b_home = QPushButton("Home"); b_home.clicked.connect(guard(on_home))
    row2.addWidget(b_set); row2.addWidget(b_cap); row2.addWidget(b_home)
    row2.addStretch()
    root.addLayout(row2)

    # --- plate mapping: teach corners -> similarity fit ---
    cal_lbl = QLabel("plate: no corners taught")
    cal_lbl.setStyleSheet(f"color:{MUTED};")

    def on_teach(corner):
        x, y, z, r = arm.pose()
        plate.teach(corner, (x, y))
        rpt = plate.report()
        cal_lbl.setText("plate: " + rpt)
        good = ("residual" in rpt and plate.res_max <= 0.5)
        cal_lbl.setStyleSheet(f"color:{ACCENT if good else MUTED};")
        print(f"[bench] taught {corner} = ({x:.1f}, {y:.1f}); {rpt}")

    row3 = QHBoxLayout()
    pitch_e = QLineEdit(str(DEFAULT_PITCH)); pitch_e.setFixedWidth(55)
    pitch_e.editingFinished.connect(guard(
        lambda: setattr(plate, "pitch", float(pitch_e.text() or DEFAULT_PITCH)) or
        setattr(plate, "fitted", False)))
    row3.addWidget(QLabel("Pitch:")); row3.addWidget(pitch_e)
    for corner in ("A1", "A6", "D1", "D6"):
        b = QPushButton(f"Teach {corner}")
        b.clicked.connect(guard(lambda c=corner: on_teach(c)))
        row3.addWidget(b)
    row3.addStretch()
    root.addLayout(row3)
    root.addWidget(cal_lbl)

    # --- go to a well center ---
    def on_goto():
        name = well_sel.currentText()
        x, y = plate.well_xy(name)                # raises clearly if not fit yet
        z = float(wellz.text() or 0)
        run_bg(lambda: arm.move_to(x, y, z, wait=True),
               f"go to {name} → ({x:.1f}, {y:.1f})")

    row4 = QHBoxLayout()
    wellz = QLineEdit(str(arm.home_xyz[2])); wellz.setFixedWidth(60)
    well_sel = QComboBox(); well_sel.addItems(all_wells())
    b_goto = QPushButton("Go to well"); b_goto.clicked.connect(guard(on_goto))
    row4.addWidget(QLabel("Well Z:")); row4.addWidget(wellz)
    row4.addWidget(well_sel); row4.addWidget(b_goto)
    row4.addStretch()
    root.addLayout(row4)

    # ================= PIERCE (built on the frozen move_to/pose) =============
    # Absolute Z setpoints, captured by feel: jog (or hand-guide) the needle to
    # the spot, then Capture. pierce_z (k) = down/inject; reset_z (j) = up/out.
    zst = {"pierce": 20.0, "reset": float(arm.home_xyz[2])}

    # --- jog row (needed for software positioning before Capture) ---
    def do_jog(dz=0.0, dx=0.0, dy=0.0):
        x, y, z, r = arm.pose()
        run_bg(lambda: arm.move_to(x + dx, y + dy, z + dz, wait=True),
               f"jog Δ({dx:+.0f},{dy:+.0f},{dz:+.0f})")

    row5 = QHBoxLayout()
    jstep = QLineEdit("2.0"); jstep.setFixedWidth(48)
    row5.addWidget(QLabel("Jog step:")); row5.addWidget(jstep)

    def mk_jog(label, **d):
        b = QPushButton(label)
        b.clicked.connect(guard(lambda: do_jog(
            **{k: v * float(jstep.text() or 0) for k, v in d.items()})))
        return b
    for wdg in (mk_jog("X-", dx=-1), mk_jog("X+", dx=1),
                mk_jog("Y-", dy=-1), mk_jog("Y+", dy=1),
                mk_jog("Z-", dz=-1), mk_jog("Z+", dz=1)):
        row5.addWidget(wdg)
    row5.addStretch()
    root.addLayout(row5)

    # --- depth capture row ---
    row6 = QHBoxLayout()
    pierce_e = QLineEdit(str(zst["pierce"])); pierce_e.setFixedWidth(60)
    reset_e = QLineEdit(str(zst["reset"])); reset_e.setFixedWidth(60)
    pierce_e.editingFinished.connect(guard(
        lambda: zst.__setitem__("pierce", float(pierce_e.text() or 0))))
    reset_e.editingFinished.connect(guard(
        lambda: zst.__setitem__("reset", float(reset_e.text() or 0))))

    def cap_pierce():
        z = round(arm.pose()[2], 1); zst["pierce"] = z; pierce_e.setText(str(z))
        status.setText(f"pierce Z captured: {z}")

    def cap_reset():
        z = round(arm.pose()[2], 1); zst["reset"] = z; reset_e.setText(str(z))
        status.setText(f"reset Z captured: {z}")

    row6.addWidget(QLabel("Pierce Z (k):")); row6.addWidget(pierce_e)
    b_capk = QPushButton("Capture ← current Z"); b_capk.clicked.connect(guard(cap_pierce))
    row6.addWidget(b_capk)
    row6.addWidget(QLabel("   Reset Z (j):")); row6.addWidget(reset_e)
    b_capj = QPushButton("Capture ← current Z"); b_capj.clicked.connect(guard(cap_reset))
    row6.addWidget(b_capj)
    row6.addStretch()
    root.addLayout(row6)

    # --- pierce actions ---
    def do_pierce():
        x, y, z, r = arm.pose()
        run_bg(lambda: arm.move_to(x, y, zst["pierce"], wait=True), "pierce → k")

    def do_reset():
        x, y, z, r = arm.pose()
        run_bg(lambda: arm.move_to(x, y, zst["reset"], wait=True), "reset → j")

    def do_pierce_well():
        name = well_sel.currentText()
        x, y = plate.well_xy(name)                 # XY from the frozen map

        def seq():
            arm.move_to(x, y, zst["reset"], wait=True)    # over well at reset height
            arm.move_to(x, y, zst["pierce"], wait=True)   # down to k
            time.sleep(0.3)                                # inject window (future pump)
            arm.move_to(x, y, zst["reset"], wait=True)     # back up to j
        run_bg(seq, f"pierce well {name}")

    row7 = QHBoxLayout()
    b_pierce = QPushButton("Pierce (→k)"); b_pierce.clicked.connect(guard(do_pierce))
    b_reset = QPushButton("Reset (→j)"); b_reset.clicked.connect(guard(do_reset))
    b_pwell = QPushButton("Pierce well (go→pierce→reset)")
    b_pwell.clicked.connect(guard(do_pierce_well))
    for wdg in (b_pierce, b_reset, b_pwell):
        row7.addWidget(wdg)
    row7.addStretch()
    root.addLayout(row7)

    # ================= RUN PLATE: go→pierce→reset for every well ============
    # Visiting order = all_wells() = A1..A6, B1..B6, C1..C6, D1..D6 (row-major,
    # as requested). Runs on a worker thread with a Stop flag; progress is
    # published to a dict the main-thread timer reads (never touch widgets from
    # the worker).
    run_state = {"stop": False, "active": False, "well": ""}
    prog_lbl = QLabel("")
    prog_lbl.setStyleSheet(f"color:{MUTED};")

    def do_run():
        if busy["v"] or not arm.connected:
            return
        if plate.n() < 2:
            status.setText("⚠ teach the plate corners first")
            status.setStyleSheet(f"color:{DANGER}; font-weight:600;")
            return
        plate.well_xy("A1")                       # force a fit now (raises if bad)
        run_state.update(stop=False, active=True, well="")

        def seq():
            try:
                wells = all_wells()
                n = len(wells)
                for i, name in enumerate(wells, 1):
                    if run_state["stop"]:
                        print(f"[bench] run aborted before {name}")
                        break
                    run_state["well"] = f"{name}  ({i}/{n})"
                    bx, by = plate.well_xy(name)
                    arm.move_to(bx, by, zst["reset"], wait=True)   # over well, up (j)
                    arm.move_to(bx, by, zst["pierce"], wait=True)  # down to k
                    time.sleep(float(dwell_e.text() or 0))          # inject window
                    arm.move_to(bx, by, zst["reset"], wait=True)   # back up to j
                print("[bench] run finished")
            finally:
                run_state["active"] = False
        run_bg(seq, "running plate…")

    def do_stop():
        run_state["stop"] = True
        status.setText("stopping after current well…")

    row8 = QHBoxLayout()
    dwell_e = QLineEdit("0.5"); dwell_e.setFixedWidth(48)
    b_run = QPushButton("Run plate (A1→A6, B1→B6, …)")
    b_run.clicked.connect(guard(do_run))
    b_stop = QPushButton("Stop"); b_stop.clicked.connect(guard(do_stop))
    row8.addWidget(QLabel("Dwell (s):")); row8.addWidget(dwell_e)
    row8.addWidget(b_run); row8.addWidget(b_stop); row8.addWidget(prog_lbl)
    row8.addStretch()
    root.addLayout(row8)
    root.addStretch()

    # live pose poll (paused while a threaded move runs); progress always shown.
    def tick():
        if run_state["active"]:
            prog_lbl.setText(f"running: {run_state['well']}")
            prog_lbl.setStyleSheet(f"color:{ACCENT}; font-weight:600;")
        elif prog_lbl.text().startswith("running"):
            prog_lbl.setText("run done")
            prog_lbl.setStyleSheet(f"color:{MUTED};")
        if not busy["v"] and arm.connected:
            refresh()

    timer = QTimer(win)
    timer.timeout.connect(guard(tick))
    timer.start(500)
    win._timer = timer
    win._arm = arm
    win.destroyed.connect(timer.stop)

    win.setCentralWidget(central)
    win.resize(860, 520)
    return win, arm


if __name__ == "__main__":
    import sys, os, traceback, faulthandler
    # crash-safe logging: line-buffered, teed, native aborts captured
    LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dobotbench.log")
    _logf = open(LOG, "w", buffering=1)
    faulthandler.enable(_logf)

    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, t):
            for st in self.s:
                try: st.write(t); st.flush()
                except Exception: pass
        def flush(self):
            for st in self.s:
                try: st.flush()
                except Exception: pass
    sys.stdout = _Tee(sys.__stdout__, _logf)
    sys.stderr = _Tee(sys.__stderr__, _logf)
    sys.excepthook = lambda t, e, tb: traceback.print_exception(t, e, tb)
    print(f"[bench] log at {LOG}")

    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QWidget{background:#1b1e23;color:#e6e9ef;font-family:'Segoe UI';font-size:13px;}"
        "QPushButton{background:#2c313a;border:1px solid #323843;border-radius:8px;padding:8px 12px;}"
        "QPushButton:hover{border:1px solid #1bb89a;}"
        "QLineEdit{background:#2c313a;border:1px solid #323843;border-radius:8px;padding:6px 8px;}")
    window, _arm = build_window()
    window.show()
    app.aboutToQuit.connect(lambda: (window._timer.stop(), window._arm.disconnect()))
    sys.exit(app.exec())
