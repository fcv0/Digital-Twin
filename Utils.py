import numpy as np
import struct, os, pandas, zlib, math, configparser, time, threading, queue
from Priv import _Vars


class RingBuffer:
    """List-compatible float32 ring buffer backed by a preallocated numpy array.

    Drop-in replacement for the Python lists the device dict previously used
    for per-cycle sensor data. Uses ~10x less memory than a Python list of
    equivalent floats (contiguous float32 vs boxed Python floats + list ptrs),
    eliminates the del-shift cost of the old trim path (ring wraps for free),
    and lets the CSV writer / plots slice contiguous numpy views instead of
    copying Python lists every flush.

    Interface compatibility (only the ops the existing code actually uses):
        rb.append(x)     O(1), no allocation, wraps at capacity
        rb[-1], rb[i]    O(1) logical indexing (0 = oldest, len-1 = newest)
        rb[a:b]          list slice (matches previous Python-list behaviour)
        len(rb)          logical length, capped at capacity
        list(rb)         ordered copy oldest->newest (via __iter__ over snapshot)
        for x in rb:     iterates oldest->newest
        rb.clear()       reset
        bool(rb)         True iff non-empty
        del rb[:n]       NO-OP — ring semantics already discard the oldest; kept
                         so utils.trimBuffers can still send `del rb[:n]` at us
                         without needing to know the container type.
        np.asarray(rb)   snapshot (via __array__)

    Extras used by the streaming CSV writer / memmap history writer:
        rb.total_written  monotonically-increasing append count (never wraps)
        rb.snapshot()     ordered numpy copy of current contents
        rb.tail_since(n)  numpy slice of samples appended since total_written==n
                          (returns None if the range has wrapped out of the
                          buffer, so the caller can log data loss and fall back
                          to the full snapshot).
    """

    __slots__ = ("_buf", "_cap", "_head", "_count", "total_written")

    def __init__(self, capacity):
        self._cap = max(1, int(capacity))
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._head = 0                  # index of the next write slot
        self._count = 0                 # logical length, capped at cap
        self.total_written = 0          # unbounded monotone append counter

    # --- container protocol ------------------------------------------------

    def __len__(self):
        return self._count

    def __bool__(self):
        return self._count > 0

    def _oldest_index(self):
        # Logical index 0 lives at (head - count) mod cap
        return (self._head - self._count) % self._cap

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._count)
            if step != 1 or stop <= start:
                return [self[i] for i in range(start, stop, step)] if step else []
            oldest = (self._oldest_index() + start) % self._cap
            n = stop - start
            end = oldest + n
            if end <= self._cap:
                return self._buf[oldest:end].tolist()
            return np.concatenate([self._buf[oldest:self._cap],
                                   self._buf[:end - self._cap]]).tolist()
        if key < 0:
            key += self._count
        if key < 0 or key >= self._count:
            raise IndexError("ring buffer index out of range")
        return float(self._buf[(self._oldest_index() + key) % self._cap])

    def __iter__(self):
        # Iterate the snapshot; much faster than N __getitem__ calls for list(rb).
        return iter(self.snapshot().tolist())

    def __delitem__(self, key):
        # No-op: ring buffer inherently drops the oldest sample once capacity
        # is exceeded, so utils.trimBuffers' `del rb[:n]` is redundant. Kept
        # for API compatibility so trimBuffers can walk mixed containers.
        pass

    def __array__(self, dtype=None):
        arr = self.snapshot()
        return arr.astype(dtype) if dtype is not None else arr

    # --- mutation ---------------------------------------------------------

    def append(self, value):
        self._buf[self._head] = value
        self._head = (self._head + 1) % self._cap
        if self._count < self._cap:
            self._count += 1
        self.total_written += 1

    def clear(self):
        self._head = 0
        self._count = 0
        self.total_written = 0

    # --- fast paths used by the CSV / memmap writers ----------------------

    def snapshot(self):
        """Ordered copy of current contents (oldest -> newest) as float32."""
        if self._count == 0:
            return np.empty(0, dtype=np.float32)
        if self._count < self._cap:
            # Buffer not yet full — data is contiguous from 0..count.
            return self._buf[:self._count].copy()
        # Buffer full: contents wrap from head to end, then 0 to head.
        return np.concatenate([self._buf[self._head:], self._buf[:self._head]])

    def tail_since(self, prior_total):
        """Return samples appended since total_written == prior_total.
        Returns np.empty(0) if nothing new, or None if the range has wrapped
        out of the buffer (data loss — caller should warn and best-effort)."""
        new = self.total_written - int(prior_total)
        if new <= 0:
            return np.empty(0, dtype=np.float32)
        if new > self._count:
            return None            # requested tail is older than we still hold
        return self.snapshot()[-new:]

# Structured dtype used by the packed-binary history writer. Each row is one
# float32 per column in _DATASAVE_COLUMNS order; the file is a raw sequence of
# these records with no header, so np.memmap can view arbitrarily large runs
# without pulling them into RAM. Sits alongside the CSV — CSVs stay portable
# for downstream tools; the .bin is for the in-app History tab.
_BIN_ITEMSIZE_FLOATS = None    # set once after _DATASAVE_COLUMNS is defined


_DATASAVE_COLUMNS = [
    "raw_main_0",    "raw_main_1",    "raw_main_2",
    "raw_ref_0",     "raw_ref_1",     "raw_ref_2",
    "raw_scatter_0", "raw_scatter_1", "raw_scatter_2",
    "avg_main_0",    "avg_main_1",    "avg_main_2",
    "avg_ref_0",     "avg_ref_1",     "avg_ref_2",
    "avg_scatter_0", "avg_scatter_1", "avg_scatter_2",
    "od",            "kod",
    "ml_ref_pred",   "ml_od",
    "raw_temp",      "avg_temp",
    "beta",          "relay_state",   "motor_state",
    "time_hw",
    "control_u",     "control_error",   "control_I",
    "time_sw",
]


_BIN_DTYPE = np.dtype([(c, "<f4") for c in _DATASAVE_COLUMNS])


def _bin_path_for(csv_path):
    """Sibling packed-binary path for a CSV path (`foo.csv` -> `foo.bin`)."""
    if csv_path.lower().endswith(".csv"):
        return csv_path[:-4] + ".bin"
    return csv_path + ".bin"


def load_history_memmap(port):
    """Return an np.memmap over this port's run history .bin file, or None if
    nothing's been written yet. Read-only. Structured dtype — fields match
    _DATASAVE_COLUMNS, e.g. mm['od'], mm['avg_temp'], mm['time_sw']."""
    dev = _Vars.activeDevices.get(port)
    if dev is None:
        return None
    path = dev.get("_bin_path")
    if not path:
        csv_path = dev.get("_csv_path")
        if not csv_path:
            return None
        path = _bin_path_for(csv_path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    n_rows = os.path.getsize(path) // _BIN_DTYPE.itemsize
    if n_rows == 0:
        return None
    return np.memmap(path, dtype=_BIN_DTYPE, mode="r", shape=(n_rows,))


def _device_column_arrays(dev):
    """Return the per-device list references in the exact order of _DATASAVE_COLUMNS.
    Kept here (not on utils) so both dataSave and flushCsv agree on the mapping."""
    return [
        dev["SensorsRaw"]["Main"][0],    dev["SensorsRaw"]["Main"][1],    dev["SensorsRaw"]["Main"][2],
        dev["SensorsRaw"]["Ref"][0],     dev["SensorsRaw"]["Ref"][1],     dev["SensorsRaw"]["Ref"][2],
        dev["SensorsRaw"]["Scatter"][0], dev["SensorsRaw"]["Scatter"][1], dev["SensorsRaw"]["Scatter"][2],
        dev["SensorsAvg"]["Main"][0],    dev["SensorsAvg"]["Main"][1],    dev["SensorsAvg"]["Main"][2],
        dev["SensorsAvg"]["Ref"][0],     dev["SensorsAvg"]["Ref"][1],     dev["SensorsAvg"]["Ref"][2],
        dev["SensorsAvg"]["Scatter"][0], dev["SensorsAvg"]["Scatter"][1], dev["SensorsAvg"]["Scatter"][2],
        dev["OpticalDensity"]["OD"],     dev["OpticalDensity"]["KOD"],
        dev["MLPredictions"]["RefPred"], dev["MLPredictions"]["OD"],
        dev["SensorsRaw"]["Temp"],       dev["SensorsAvg"]["Temp"],
        dev["Coefficients"]["Beta"],     dev["DeviceVariables"]["RelayState"],
        dev["DeviceVariables"]["MotorState"],
        dev["Time"]["Hardware"],
        dev["DeviceVariables"]["ControlU"],  dev["DeviceVariables"]["ControlError"],
        dev["DeviceVariables"]["ControlI"],
        dev["Time"]["Software"],
    ]

class utils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def readConfig() -> None:
        # inline_comment_prefixes lets values carry a trailing "; comment" without it
        # becoming part of the value (e.g. Port = auto ; try every port).
        cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        # Reactor config + simulator config are separate files; sections are
        # disjoint so a single parser can hold both.
        cfg_path = utils.config_paths()
        cfg.read(cfg_path)
        _Vars.config = {
            "MotorSpeeds":       [int(x)   for x in cfg.get("DeviceVariables", "MotorSpeeds").split(",")],
            "MotorDirections":   [int(x)   for x in cfg.get("DeviceVariables", "MotorDirections").split(",")],
            "CH0Speed":          cfg.getint("DeviceVariables", "CH0Speed", fallback=0),
            "Turbidostat":       cfg.getboolean("DeviceVariables", "Turbidostat"),
            "TurbidostatMode":   cfg.get("DeviceVariables", "TurbidostatMode"),
            "HeaterMode":        cfg.get("DeviceVariables", "HeaterMode"),
            "SendMessage":       cfg.getboolean("DeviceVariables", "SendMessage"),
            "Temperature":       [float(x) for x in cfg.get("ControlValues", "Temperature").split(",")],
            "OD":                [float(x) for x in cfg.get("ControlValues", "OD").split(",")],
            "CalibrationConst":  cfg.getfloat("Offsets", "CalibrationConst"),
            "TemperatureOffset": cfg.getfloat("Offsets", "TemperatureOffset"),
            "Beta":              cfg.getfloat("Calibration", "Beta",       fallback=0.0),
            "BetaSource":        cfg.get("Calibration",      "BetaSource", fallback="auto"),
            "Strain":            cfg.get("Run",     "Strain",        fallback="SB5").strip(),
            "SaveDirectory":     cfg.get("Storage", "SaveDirectory", fallback="").strip(),
            "BetaTime":          cfg.getint("Timers", "BetaTime"),
            "Multiplex":         cfg.getboolean("Network", "Multiplex"),
            "Port":              cfg.getint("Network", "Port"),
            "AutoIP":            cfg.getboolean("Network", "AutoIP", fallback=True),
            "Offline":           cfg.getboolean("Debug", "Offline", fallback=False),
            "OfflinePorts":      cfg.getint("Debug", "OfflinePorts", fallback=2),
            "TimeScale":         cfg.getfloat("Debug", "TimeScale",  fallback=1.0),
            "Twin":              cfg.getboolean("Debug", "Twin", fallback=False),
            # Digital-twin growth knobs (demo-tuned overrides of the physical defaults).
            "TwinMuMean":        cfg.getfloat("Twin", "MuMean",           fallback=0.002),
            "TwinInoculationS":  cfg.getfloat("Twin", "InoculationS",     fallback=120.0),
            "TwinLagS":          cfg.getfloat("Twin", "LagS",             fallback=300.0),
            "TwinCarryingCap":   cfg.getfloat("Twin", "CarryingCapacity", fallback=1e9),
            "TwinAlphaAtt":      cfg.getfloat("Twin", "AlphaAtt",         fallback=2.5e-9),
            "TwinBubbles":       cfg.getboolean("Twin", "Bubbles",        fallback=True),
            "TwinBubbleMicroRate": cfg.getfloat("Twin", "BubbleMicroRate", fallback=0.09),
            "TwinBubbleLargeRate": cfg.getfloat("Twin", "BubbleLargeRate", fallback=0.0004),
            "TwinBubbleChurnRate": cfg.getfloat("Twin", "BubbleChurnRate", fallback=1.0 / 15000.0),
            "TwinTurbidostat":   cfg.getboolean("Twin", "Turbidostat",    fallback=True),
            "TwinPumpFlow":      cfg.getfloat("Twin", "PumpFlow",         fallback=0.5),
            "TwinVesselVolume":  cfg.getfloat("Twin", "VesselVolume",     fallback=50.0),
            "TwinTransportLagS": cfg.getfloat("Twin", "TransportLagS",    fallback=4.0),
            "TwinMediaTemp":     cfg.getfloat("Twin", "MediaTemp",        fallback=25.0),
            "TwinCircThreshold": cfg.getfloat("Twin", "CirculationThreshold", fallback=50.0),
            "SerialProvision":   cfg.getboolean("Serial", "Provision", fallback=False),
            "SerialPort":        cfg.get("Serial", "Port", fallback=""),
            "SerialBaud":        cfg.getint("Serial", "Baud", fallback=115200),
            "TempPI_Kp":         cfg.getfloat("TempPI", "Kp"),
            "TempPI_Ki":         cfg.getfloat("TempPI", "Ki"),
            "KalmanModel":       cfg.get("Kalman", "Model", fallback="scalar"),
            "DespikeHoldoffS":   cfg.getfloat("Kalman", "DespikeHoldoffS", fallback=20.0),
            "ControlUsesFilteredOD": cfg.getboolean("Kalman", "ControlUsesFilteredOD", fallback=True),
            "MicroscopeEnabled": cfg.getboolean("Microscope", "Enabled",     fallback=False),
            "DobotEnabled":      cfg.getboolean("Dobot",      "Enabled",     fallback=False),
            "MicroscopeExposureMs": cfg.getint("Microscope", "ExposureMs",   fallback=50),
            "MicroscopeLiveIntervalS": cfg.getfloat("Microscope", "LiveIntervalS", fallback=0.15),
            "WindowSeconds":     cfg.getint("Memory", "WindowSeconds",       fallback=21600),
            "FlushIntervalSeconds": cfg.getint("Memory", "FlushIntervalSeconds", fallback=5),
        }
        _Vars.isConfig = True
        print(f"[INFO] Config loaded from {cfg_path}")

    @staticmethod
    def config_dir() -> str:
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), "config")

    # Reactor/hardware settings vs simulator-only settings, kept in separate files
    # so an operator never has to scroll past twin knobs to change a setpoint.
    CONFIG_FILES = ("config.ini", "simulator.ini")

    @staticmethod
    def config_paths() -> list:
        return [os.path.join(utils.config_dir(), n) for n in utils.CONFIG_FILES]

    @staticmethod
    def config_path() -> str:
        """The main reactor config. Kept as the default target for writes whose
        section isn't found in any file."""
        return os.path.join(utils.config_dir(), "config.ini")

    @staticmethod
    def _section_file_map() -> dict:
        """section (lowercased) -> path of the ini file that declares it, so a
        save routes each value back to the file it came from."""
        m = {}
        for path in utils.config_paths():
            if not os.path.exists(path):
                continue
            cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
            cp.optionxform = str
            cp.read(path)
            for s in cp.sections():
                m[s.lower()] = path
        return m

    @staticmethod
    def readConfigItems():
        """Ordered [(source_name, section, key, value_str), ...] across every
        config file, with inline comments stripped. Preserves declaration order
        and key case (unlike the default configparser which lowercases keys) so
        the Config tab can render fields that map 1:1 back to their file."""
        items = []
        for path in utils.config_paths():
            if not os.path.exists(path):
                continue
            cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
            cfg.optionxform = str                  # keep MotorSpeeds, not motorspeeds
            cfg.read(path)
            src = os.path.basename(path)
            for section in cfg.sections():
                for key, value in cfg.items(section):
                    items.append((src, section, key, value))
        return items

    @staticmethod
    def saveConfigValues(updates) -> int:
        """Write {(section, key): value} back to disk, routing each value to the
        config file that declares its section (reactor settings -> config.ini,
        simulator/twin settings -> simulator.ini). Sections not found in any file
        fall back to the main config. Returns the number of values written."""
        smap = utils._section_file_map()
        by_file = {}
        for (sec, key), val in updates.items():
            path = smap.get(sec.lower(), utils.config_path())
            by_file.setdefault(path, {})[(sec, key)] = val
        return sum(utils._save_values_to(path, ups) for path, ups in by_file.items())

    @staticmethod
    def _save_values_to(path, updates) -> int:
        """Write {(section, key): value} into one ini file in place, preserving
        comments, blank lines, section order, and any inline `; comment` on each
        line. Returns the number of values written. Keys already present are
        edited in place; keys missing from the file are appended to their section
        (a missing section is created at the end).

        configparser.write() would drop every comment in the file, so we do a
        line-based rewrite instead — these configs are hand-maintained with
        helpful comments we don't want to lose."""
        import re
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Normalise keys → dict[(section_lower, key_lower)] = (orig_section, orig_key, value)
        pending = {}
        for (sec, key), val in updates.items():
            pending[(sec.lower(), key.lower())] = (sec, key, str(val))

        out = []
        cur_section = None
        applied = set()
        sec_re = re.compile(r'^\s*\[(?P<name>[^\]]+)\]\s*$')
        kv_re  = re.compile(r'^(?P<indent>\s*)(?P<key>[^=\s;#][^=]*?)(?P<sp>\s*=\s*)'
                            r'(?P<val>.*?)(?P<cmt>\s+[;#].*)?\s*$')

        for line in lines:
            m_sec = sec_re.match(line)
            if m_sec:
                cur_section = m_sec.group("name").strip()
                out.append(line)
                continue
            m_kv = kv_re.match(line)
            if m_kv and cur_section is not None:
                key = m_kv.group("key").strip()
                pk = (cur_section.lower(), key.lower())
                if pk in pending:
                    _, _, new_val = pending[pk]
                    cmt = m_kv.group("cmt") or ""
                    nl = "\n" if line.endswith("\n") else ""
                    out.append(f"{m_kv.group('indent')}{key}{m_kv.group('sp')}{new_val}{cmt}{nl}")
                    applied.add(pk)
                    continue
            out.append(line)

        # Append any updates whose key wasn't found in the file.
        missing = [(pk, v) for pk, v in pending.items() if pk not in applied]
        if missing:
            # Group missing keys by (original) section.
            existing_sections = {m.group("name").strip().lower()
                                 for m in (sec_re.match(l) for l in out) if m}
            by_section = {}
            for pk, (sec, key, val) in missing:
                by_section.setdefault(sec, []).append((key, val))
            for sec, kvs in by_section.items():
                if sec.lower() not in existing_sections:
                    if out and out[-1].strip() != "":
                        out.append("\n")
                    out.append(f"[{sec}]\n")
                    for key, val in kvs:
                        out.append(f"{key} = {val}\n")
                else:
                    # Insert after the last line of that section.
                    insert_at = len(out)
                    in_target = False
                    for i, line in enumerate(out):
                        ms = sec_re.match(line)
                        if ms:
                            in_target = (ms.group("name").strip().lower() == sec.lower())
                        elif in_target and line.strip() == "":
                            insert_at = i
                    add = [f"{key} = {val}\n" for key, val in kvs]
                    out[insert_at:insert_at] = add

        with open(path, "w", encoding="utf-8", newline="") as f:
            f.writelines(out)
        return len(pending)

    @staticmethod
    def resetDevice(port) -> None:
        device = _Vars.activeDevices[port]
        # Flush anything the periodic writer hasn't yet drained, so Reset doesn't
        # discard the last few unflushed seconds. Silent on error — reset should
        # never fail because of a disk hiccup.
        try:
            utils.flushCsv(port)
        except Exception as e:
            print(f"[WARN] Pre-reset flush on port {port} failed: {e}")
        for section in ("SensorsRaw", "SensorsAvg", "BufferArrs"):
            for key in ("Main", "Ref", "Scatter"):
                for arr in device[section][key]:
                    arr.clear()
            device[section]["Temp"].clear()
        device["OpticalDensity"]["OD"].clear()
        device["OpticalDensity"]["KOD"].clear()
        device["OpticalDensity"]["EstimatedDt"].clear()
        device["MLPredictions"]["RefPred"].clear()
        device["MLPredictions"]["OD"].clear()
        device["DeviceVariables"]["RelayState"].clear()
        device["DeviceVariables"]["MotorState"].clear()
        device["DeviceVariables"]["ControlU"].clear()
        device["DeviceVariables"]["ControlError"].clear()
        device["DeviceVariables"]["ControlI"].clear()
        # Auto beta is per-run — clear it so the next run re-learns the blank.
        # Predefined beta + source are config-driven and left untouched.
        device["Coefficients"]["Beta"].clear()
        device["Coefficients"]["BetaFinal"] = 0.0
        device["Time"]["Software"].clear()
        device["Time"]["Hardware"].clear()
        device["corrected_OD"].clear()
        device["_initTime"] = time.time()
        device["DeviceVariables"]["bb_target_hit_time"] = None
        if "ML" in device and "Kalman" in device["ML"]:
            device["ML"]["Kalman"].reset()
        # Start a new run CSV — the old file stays on disk, next flush opens a fresh one.
        utils.resetRunCsv(port)
        print(f"[INFO] Device {port} reset — all arrays cleared, BetaTime restarted, new CSV on next flush.")

    @staticmethod
    def convertFloating2Bytes(value) -> bytearray:
        try:        return struct.pack("f", value)
        except Exception as e:
            print(f"[WARN] convertFloating2Bytes({value!r}): {e}")
            return bytes([0x00, 0x00, 0x00, 0x00])
    
    @staticmethod
    def twoByteConv(value, ) -> bytes:
        return int.to_bytes(value, length=2, byteorder="little")

    @staticmethod
    def checkInteger(value) -> tuple:
        # Returns (direction, speed). Leading '-' = reverse; "0"/"" = stop.
        value = value.strip()
        if value.startswith("-"):
            dir = 1
            value = value[1:]
        else:
            dir = 0
        speed = int(value) if value else 0
        return dir, speed

    @staticmethod
    def packFloatingBytes(fBytes, idx, port) -> None:
        _Vars.activeDevices[port]["OutgoingMessage"][idx], _Vars.activeDevices[port]["OutgoingMessage"][idx+1], _Vars.activeDevices[port]["OutgoingMessage"][idx+2], _Vars.activeDevices[port]["OutgoingMessage"][idx+3] = fBytes[0], fBytes[1], fBytes[2], fBytes[3]

    @staticmethod
    def unpackFloatingFromBytes(fBytes) -> float:
        return struct.unpack("f", fBytes)[0]
    
    @staticmethod
    def unpackIntFromBytes(byte) -> int:
        return int.from_bytes(byte, "little")     
    
    @staticmethod
    def changeMotorSpeeds(text: str, idx: list, sPosition: int, port) -> None:
        '''
        text = str(value) = 0-255/0-4095 can include -.
        idx[0] = first byte position in out-going message.
        idx[1] = last byte position in out-going message.
        ~ idx[2] = can be an input given highRes is selected.
        sPosition = configurational positon of the motor speed and/or direction [x, y] -> _Vars.motorDirections, _Vars.motorSpeeds.
        - Saves state of speed given incorrect text inputs.
        '''
        if len(text) == 0:
            text = "0"
            _Vars.activeDevices[port]["DeviceVariables"]["MotorDirections"][sPosition[0]], _Vars.activeDevices[port]["DeviceVariables"]["MotorSpeeds"][sPosition[1]] = 0, 0
        dir, speed = 0, 0
        try:
            dir, speed = utils.checkInteger(text)
            _Vars.activeDevices[port]["DeviceVariables"]["MotorSpeeds"][sPosition[1]] = speed
            if idx[0] == -1:
                _Vars.activeDevices[port]["DeviceVariables"]["MotorDirections"][sPosition[0]] = 0
                pass
            else:
                _Vars.activeDevices[port]["DeviceVariables"]["MotorDirections"][sPosition[0]] = dir
                _Vars.activeDevices[port]["OutgoingMessage"][idx[0]] = dir
            if len(idx) > 2: # essentially if high resolution motors
                b1, b2 = utils.twoByteConv(speed)
                _Vars.activeDevices[port]["OutgoingMessage"][idx[1]:idx[2]] = b1, b2
            else:
                _Vars.activeDevices[port]["OutgoingMessage"][idx[1]] = speed
        except Exception as e:
            print(f"[ERROR] | MOTOR CONTROL ERROR: {e}")
        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True
    
    @staticmethod
    def changeCH0Speed(text, port) -> None:
        """Auxiliary CH0 motor: a single always-on speed (0-255) packed into
        byte 53. This is a STEPPER/peristaltic channel (PUMP_CH0), so the value
        is an AD5245 digipot level, 255 = fastest — NOT a percent like the
        circulation/aeration fans. Keeps its previous value on partial/invalid
        input, like the other motor fields."""
        try:
            v = max(0, min(255, int(float(text)))) if str(text).strip() else 0
            _Vars.activeDevices[port]["DeviceVariables"]["CH0Speed"] = v
            _Vars.activeDevices[port]["OutgoingMessage"][53] = v
            _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True
        except (ValueError, TypeError):
            pass   # partial input ("" / "-") — hold the previous value

    @staticmethod
    def changeControlValues(text, idx, sPosition, controlSelection, port) -> None:
        '''
        idx[0] = first byte position in out-going message.
        idx[1] = last byte position in out-going message.
        sPosition = stores the current stored control values (save-point)
        - Saves its own state given incorrect inputs.
        '''
        try:
            value = float(text)
            _Vars.activeDevices[port]["DeviceVariables"]["ControlValues"][controlSelection][sPosition] = value
            messageBytes = utils.convertFloating2Bytes(value)
            utils.packFloatingBytes(messageBytes, idx=idx, port=port)
        except (ValueError, TypeError):
            # Expected on partial input (e.g. "" or "-") — keep the previous value.
            messageBytes = utils.convertFloating2Bytes(_Vars.activeDevices[port]["DeviceVariables"]["ControlValues"][controlSelection][sPosition])
            utils.packFloatingBytes(messageBytes, idx=idx, port=port)
        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True

    @staticmethod
    def addTempOffset(text, port) -> None:
        try:
            _Vars.activeDevices[port]["Coefficients"]["TemperatureOffset"] = float(text)
            float_bytes = utils.convertFloating2Bytes(float(text))
            utils.packFloatingBytes(float_bytes, 10, port=port)  # temp_offset at bytes 10-13
            _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True
        except (ValueError, TypeError):
            # Partial input ("" / "-") — keep the previous offset, don't repack.
            pass

    @staticmethod
    def ODHyst(currentOD: float, bandwidth: float, port: int, targetOD: float):
        # Drives dilution via the OD trigger byte (33), consumed by od_pump_control
        # mode 1. Hysteresis hold: between (target-bandwidth) and target, msg[33]
        # is left unchanged so the pump keeps its previous state.
        msg = _Vars.activeDevices[port]["OutgoingMessage"]
        if currentOD >= targetOD:
            msg[33] = 255   # dilute
            # Mark a real dose: the despiker must stand down while the vessel
            # actually dilutes (see Comms._despike_allowed).
            _Vars.activeDevices[port]["_pump_last_active"] = time.time()
        elif currentOD <= (targetOD - bandwidth):
            msg[33] = 127   # stop
        # MotorState is logged centrally per-cycle in Comms.listenPorts (_pump_state).
        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True

    @staticmethod
    def bangbang(currentOD: float, targetOD: float, port: int):
        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True
        msg = _Vars.activeDevices[port]["OutgoingMessage"]
        if currentOD >= targetOD:
            msg[33] = 255  # OD pump trigger / ss1
            # Mark a real dose: the despiker must stand down while the vessel
            # actually dilutes (see Comms._despike_allowed).
            _Vars.activeDevices[port]["_pump_last_active"] = time.time()
        else:
            msg[33] = 127
        
    @staticmethod
    def calcBeta(port) -> None:
        """Always compute the AUTO blank ratio beta = Main/Ref (no cells) as an
        output: accumulate it during the first BetaTime seconds of the run, then
        latch BetaFinal = mean once BetaTime elapses. This runs every cycle
        regardless of which beta the OD calc is actually using — so the auto
        value is always available to view / save as a predefined calibration.
        """
        dev  = _Vars.activeDevices[port]
        co   = dev["Coefficients"]
        main = dev["SensorsAvg"]["Main"][1][-1]
        ref  = dev["SensorsAvg"]["Ref"][1][-1]
        if dev["Time"]["Software"][-1] <= dev["Time"]["BetaTime"]:
            if ref:
                co["Beta"].append(main / ref)
        elif not co["BetaFinal"] and co["Beta"]:
            co["BetaFinal"] = float(np.mean(co["Beta"]))

    @staticmethod
    def betaAuto(port) -> float:
        """The auto-detected beta output: latched BetaFinal if available, else the
        running mean of the blank window so far, else NaN (nothing learned yet)."""
        co = _Vars.activeDevices[port]["Coefficients"]
        if co["BetaFinal"]:
            return float(co["BetaFinal"])
        if co["Beta"]:
            return float(np.mean(co["Beta"]))
        return float("nan")

    @staticmethod
    def betaValue(port) -> float:
        """The beta the OD calc should USE, per the device's BetaSource:
          - 'predefined' + a configured value > 0  → that fixed value
          - otherwise                              → the auto-detected beta
                                                     (falls back to 1.0 if none yet)
        """
        co = _Vars.activeDevices[port]["Coefficients"]
        if co.get("BetaSource") == "predefined" and co.get("BetaPredefined", 0) > 0:
            return float(co["BetaPredefined"])
        auto = utils.betaAuto(port)
        return auto if auto == auto else 1.0        # NaN-safe
    
    @staticmethod
    def movingAvg(value, bufferLoc, port):
        if len(bufferLoc) >= _Vars.activeDevices[port]["BufferArrs"]["BufferLength"]:
            bufferLoc.pop(0)
        bufferLoc.append(value)
        return np.mean(bufferLoc)
    
    @staticmethod
    def calcOD(port) -> float:
        # OD = C · log10( (Ref · beta) / Main ), beta chosen by betaValue():
        # calibrated/latched BetaFinal, else the running blank mean, else 1.0.
        dev  = _Vars.activeDevices[port]
        C    = dev["Coefficients"]["CalibrationConst"]
        ref  = dev["SensorsAvg"]["Ref"][1][-1]
        main = dev["SensorsAvg"]["Main"][1][-1]
        beta = utils.betaValue(port)
        if not main:
            return 0.0
        return C * np.log10((ref * beta) / main)
    
    @staticmethod
    def calcSimpOD(port) -> float:
        if len(_Vars.activeDevices[port]["SensorsAvg"]["Main"][1]) < 100:
            iref = np.mean(_Vars.activeDevices[port]["SensorsAvg"]["Main"][1][:len(_Vars.activeDevices[port]["SensorsAvg"]["Main"][1])])
        else:
            iref = np.mean(_Vars.activeDevices[port]["SensorsAvg"]["Main"][1][:100])
        return _Vars.activeDevices[port]["Coefficients"]["CalibrationConst"]*np.log10(iref/_Vars.activeDevices[port]["SensorsAvg"]["Main"][1][-1])

    @staticmethod
    def pointDir() -> None:
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

    @staticmethod
    def default_log_root() -> str:
        """Fallback log root: a 'logs' folder next to the application."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

    @staticmethod
    def log_root() -> str:
        """Configured save location ([Storage] SaveDirectory), else the default.
        Falls back to the default if the configured path can't be created — a bad
        path in the config must never take the run down mid-experiment."""
        base = (_Vars.config.get("SaveDirectory") or "").strip()
        if not base:
            return utils.default_log_root()
        try:
            os.makedirs(base, exist_ok=True)
            return base
        except Exception as e:
            print(f"[WARN] SaveDirectory {base!r} unusable ({e}); "
                  f"falling back to {utils.default_log_root()}")
            return utils.default_log_root()

    @staticmethod
    def create_day_folder() -> str:
        folder_path = os.path.join(utils.log_root(), f"{_Vars.date} experiments")
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    @staticmethod
    def strain_for(port) -> str:
        """Per-reactor strain label, falling back to the configured default.
        Per-device because a multiplexer can run a different strain per vessel."""
        dev = _Vars.activeDevices.get(port)
        if dev and dev.get("Strain"):
            return str(dev["Strain"])
        return str(_Vars.config.get("Strain") or getattr(_Vars, "strain", "") or "strain")

    @staticmethod
    def safe_label(text, fallback="strain") -> str:
        """Make free text safe for a filename: keep alnum/dash/underscore/dot,
        collapse everything else to '-', and bound the length."""
        cleaned = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(text).strip())
        cleaned = "-".join(p for p in cleaned.split("-") if p)     # collapse runs of '-'
        return cleaned[:40] or fallback

    @staticmethod
    def _run_stem(port) -> str:
        """Stable per-run file stem for one port. Same across the whole run so the
        streaming CSV / XML sidecar keep the same name (no more one-file-per-save).
        The strain is captured when the run file is first opened — editing it later
        updates the XML sidecar; the CSV keeps the name it was created with until
        the run is reset."""
        return (
            f"log_{_Vars.current_datetime}"
            f"_device-{str(port)[-1]}"
            f"_{utils.safe_label(utils.strain_for(port))}"
            f"_200mL_res_clamped_2xyt"
        )

    @staticmethod
    def _open_run_csv(port):
        """Lazily bind a single CSV path to the device for this run. Idempotent —
        subsequent calls reuse whatever's already on the device dict."""
        dev = _Vars.activeDevices[port]
        if dev.get("_csv_path"):
            return dev["_csv_path"]
        utils.pointDir()
        day_dir = utils.create_day_folder()
        stem = utils._run_stem(port)
        path = os.path.join(day_dir, stem + ".csv")
        dev["_csv_path"] = path
        dev["_csv_stem"] = stem
        dev["_csv_day_dir"] = day_dir
        # If a file with this stem already exists (rare — e.g. restart within the
        # same minute), keep appending to it; header only if it's actually empty.
        dev["_csv_header_written"] = os.path.exists(path) and os.path.getsize(path) > 0
        # Ring-buffer world: track flush progress by the master ring's monotone
        # total_written counter (unbounded), NOT by array indices (which wrap).
        # Beta / legacy-list columns get their own per-name last-flushed-length
        # counters in _csv_next_col_len.
        dev.setdefault("_csv_next_total", 0)
        dev.setdefault("_csv_next_col_len", {})
        return path

    @staticmethod
    def flushCsv(port) -> int:
        """Append everything appended since the last flush to this port's run
        CSV. Returns the number of rows written.

        Design:
          - Ring-buffer columns (Time.Software, SensorsRaw/Avg, OD, KOD,
            RelayState, MotorState, corrected_OD, MLPredictions) tail off the
            master ring's monotone total_written counter.
          - Plain-list columns (Beta) tail off their own last-flushed length.
          - Every column is NaN-padded to the master flush length so partially
            populated columns don't truncate the row count.
          - Runs on the port's comms thread (the only mutator of the device
            arrays), so no lock is needed.
        """
        dev = _Vars.activeDevices.get(port)
        if dev is None:
            return 0
        utils._open_run_csv(port)

        master = dev["Time"]["Software"]              # RingBuffer
        master_total = master.total_written
        prior_total = int(dev.get("_csv_next_total", 0))
        flush_len = master_total - prior_total
        if flush_len <= 0:
            return 0

        per_col_len = dev.setdefault("_csv_next_col_len", {})
        arrays = _device_column_arrays(dev)
        tail = {}
        dropped_any = False
        for col, arr in zip(_DATASAVE_COLUMNS, arrays):
            if isinstance(arr, RingBuffer):
                piece = arr.tail_since(prior_total)
                if piece is None:               # ring wrapped past prior_total
                    dropped_any = True
                    piece = arr.snapshot()       # best effort — take whatever remains
            else:
                # Legacy list column (Beta, EstimatedDt). Its len is not master's.
                start = per_col_len.get(col, 0)
                piece = np.asarray(arr[start:], dtype=np.float32)
                per_col_len[col] = len(arr)
            # NaN-pad short columns so rows line up with the master flush length.
            if len(piece) < flush_len:
                pad = np.full(flush_len - len(piece), np.nan, dtype=np.float32)
                piece = np.concatenate([piece, pad])
            elif len(piece) > flush_len:
                # Only reachable if a column raced ahead of master, which
                # shouldn't happen (all RB columns are appended in lockstep
                # by listenPorts). Truncate defensively.
                piece = piece[-flush_len:]
            tail[col] = piece

        if dropped_any:
            print(f"[WARN] flushCsv port {port}: ring buffer wrapped before flush; "
                  f"some rows overwritten. Consider a smaller FlushIntervalSeconds "
                  f"or larger WindowSeconds.")

        df = pandas.DataFrame(tail)
        write_header = not dev.get("_csv_header_written", False)
        df.to_csv(dev["_csv_path"], mode="a", header=write_header, index=False)
        dev["_csv_header_written"] = True

        # Also append the same tail to the sibling packed-binary file, so the
        # History tab can np.memmap it later. One structured-array to_bytes call
        # per flush; ~2.5x smaller than the CSV and O(1) to random-access.
        try:
            bin_path = dev.get("_bin_path") or _bin_path_for(dev["_csv_path"])
            dev["_bin_path"] = bin_path
            rec = np.empty(flush_len, dtype=_BIN_DTYPE)
            for col in _DATASAVE_COLUMNS:
                rec[col] = tail[col]
            with open(bin_path, "ab") as bf:
                bf.write(rec.tobytes())
        except Exception as e:
            print(f"[WARN] history .bin write failed for port {port}: {e}")

        dev["_csv_next_total"] = master_total
        return int(flush_len)

    @staticmethod
    def _trimmable_arrays(dev):
        """Every growable per-cycle list on the device (Beta is intentionally omitted —
        it only grows during BetaTime and stays small). Used by trimBuffers."""
        arrs = []
        for group in ("SensorsRaw", "SensorsAvg"):
            for key in ("Main", "Ref", "Scatter"):
                arrs.extend(dev[group][key])           # 3 channels each
            arrs.append(dev[group]["Temp"])
        arrs.extend([dev["OpticalDensity"]["OD"],
                     dev["OpticalDensity"]["KOD"],
                     dev["OpticalDensity"]["EstimatedDt"]])
        arrs.extend([dev["MLPredictions"]["RefPred"],
                     dev["MLPredictions"]["OD"]])
        arrs.extend([dev["DeviceVariables"]["RelayState"],
                     dev["DeviceVariables"]["MotorState"],
                     dev["DeviceVariables"]["ControlU"],
                     dev["DeviceVariables"]["ControlError"],
                     dev["DeviceVariables"]["ControlI"]])
        arrs.extend([dev["Time"]["Software"],
                     dev["Time"]["Hardware"]])
        arrs.append(dev["corrected_OD"])
        return arrs

    @staticmethod
    def trimBuffers(port, keep_last=None) -> None:
        """No-op for RingBuffer columns (they wrap for free) — still trims any
        legacy list columns (Beta, EstimatedDt) so an errant unbounded appender
        can't leak memory. Retained so Comms.main can keep its flush+trim
        cadence without needing to know the container type.
        """
        dev = _Vars.activeDevices.get(port)
        if dev is None:
            return
        if keep_last is None:
            keep_last = int(_Vars.config.get("WindowSeconds", 21600))
        for arr in utils._trimmable_arrays(dev):
            if isinstance(arr, RingBuffer):
                continue           # ring wraps automatically at capacity
            if len(arr) > keep_last:
                del arr[: len(arr) - keep_last]

    @staticmethod
    def resetRunCsv(port) -> None:
        """Called on Reset Data: drop the current CSV/XML binding so the next
        flush opens a fresh file (new timestamp stem). The old file on disk is
        left in place — nothing is deleted."""
        dev = _Vars.activeDevices.get(port)
        if dev is None:
            return
        for k in ("_csv_path", "_csv_stem", "_csv_day_dir", "_csv_header_written",
                  "_bin_path"):
            dev.pop(k, None)
        dev["_csv_next_total"] = 0
        dev["_csv_next_col_len"] = {}

    @staticmethod
    def dataSave(*arrays, port) -> None:
        """Kept for API compatibility with existing callers (Save Data button,
        closeEvent). The *arrays payload is ignored — we now stream-append directly
        from the device dict via flushCsv, then rewrite the XML sidecar so the
        latest metadata always matches the current CSV."""
        utils.pointDir()
        n = utils.flushCsv(port)
        dev = _Vars.activeDevices[port]
        stem = dev.get("_csv_stem") or utils._run_stem(port)
        day_dir = dev.get("_csv_day_dir") or utils.create_day_folder()
        utils.saveMetadata(port, day_dir, stem)
        print(f"[INFO] Save port {port}: flushed {n} new row(s) -> {os.path.basename(dev.get('_csv_path', ''))}")

    @staticmethod
    def saveMetadata(port, day_dir, stem) -> None:
        """Write an <stem>.xml sidecar capturing every run/control variable for this
        reactor at save time: heater & OD modes, targets, hysteresis bands, bang-bang
        delays, offsets, motor (cycle/stirrer/pump) speeds + directions, calibration,
        beta, Kalman model + Q/R, LED/averaging, and network info."""
        import xml.etree.ElementTree as ET
        try:
            dev = _Vars.activeDevices[port]
            dv  = dev["DeviceVariables"]
            cv  = dv["ControlValues"]
            co  = dev["Coefficients"]
            msg = dev["OutgoingMessage"]
            cfg = _Vars.config

            def b(i):                       # one packet byte as int
                try:    return int(msg[i].item())
                except Exception: return int(msg[i])

            def put(parent, tag, val):
                e = ET.SubElement(parent, tag)
                e.text = "" if val is None else str(val)
                return e

            t_sw    = dev["Time"]["Software"]
            elapsed = t_sw[-1] if t_sw else 0
            rx, lost = dev["PacketLoss"]
            HEATER = {0: "bang-bang", 1: "hysteresis", 2: "PI"}
            ODMODE = {0: "bang-bang", 1: "hysteresis", 2: "clamp"}
            LEDM   = {0: "normal-always-on", 1: "modulated-dark-subtract"}
            temp = list(cv.get("Temperature", []))
            odv  = list(cv.get("OD", []))
            spd  = list(dv.get("MotorSpeeds", []))
            dirn = list(dv.get("MotorDirections", []))

            root = ET.Element("BioreactorRun")

            run = ET.SubElement(root, "Run")
            put(run, "GUIVersion",     getattr(_Vars, "version", ""))
            put(run, "SavedAt",        getattr(_Vars, "current_datetime", ""))
            put(run, "Date",           getattr(_Vars, "date", ""))
            put(run, "DevicePort",     port)
            put(run, "Strain",         utils.strain_for(port))
            put(run, "ElapsedSeconds", elapsed)
            put(run, "Samples",        len(dev["OpticalDensity"]["OD"]))

            net = ET.SubElement(root, "Network")
            addr = dev.get("Network", {}).get("Addr")
            put(net, "Address", addr[0] if isinstance(addr, (tuple, list)) and addr else "")
            put(net, "PacketsReceived", rx)
            put(net, "PacketsLost",     lost)
            put(net, "Multiplex",       cfg.get("Multiplex", ""))
            put(net, "AutoIP",          cfg.get("AutoIP", ""))

            th = ET.SubElement(root, "Temperature")
            put(th, "Mode",             HEATER.get(b(18), b(18)))
            put(th, "TargetC",          temp[0] if len(temp) > 0 else "")
            put(th, "HysteresisBandC",  temp[1] if len(temp) > 1 else "")
            put(th, "BangBangDelayMs",  temp[2] if len(temp) > 2 else "")
            put(th, "OffsetC",          co.get("TemperatureOffset", ""))   # authoritative offset
            put(th, "PI_Kp",            cfg.get("TempPI_Kp", ""))
            put(th, "PI_Ki",            cfg.get("TempPI_Ki", ""))

            od = ET.SubElement(root, "OpticalDensity")
            put(od, "Mode",              ODMODE.get(b(32), b(32)))
            put(od, "TurbidostatEnabled", bool(b(34)))
            put(od, "TurbidostatMode",   dv.get("TurbidostatMode", ""))
            put(od, "TargetOD",          odv[0] if len(odv) > 0 else "")
            put(od, "BangBangDelayMs",   odv[1] if len(odv) > 1 else "")
            put(od, "HysteresisBand",    odv[3] if len(odv) > 3 else "")

            mo = ET.SubElement(root, "Motors")
            # index map: 0 cycle, 1 stirrer/fan, 2 pump1 (waste), 3 pump2 (fresh)
            for i, nm in enumerate(("Cycle", "Stirrer", "Pump1", "Pump2")):
                m = ET.SubElement(mo, nm)
                put(m, "Speed",     spd[i]  if i < len(spd)  else "")
                put(m, "Direction", dirn[i] if i < len(dirn) else "")

            cal = ET.SubElement(root, "Calibration")
            put(cal, "CalibrationConst", co.get("CalibrationConst", ""))
            put(cal, "BetaFinal",        co.get("BetaFinal", ""))
            put(cal, "BetaSamples",      len(co.get("Beta", [])))
            put(cal, "BetaTimeSeconds",  dev["Time"].get("BetaTime", ""))
            put(cal, "AvgSamples",       b(52))
            put(cal, "LEDMode",          LEDM.get(b(0), b(0)))

            kf = ET.SubElement(root, "KalmanFilter")
            put(kf, "Model", cfg.get("KalmanModel", "scalar"))
            try:
                k = dev["ML"]["Kalman"]
                put(kf, "Q", getattr(k, "Q", ""))
                put(kf, "R", getattr(k, "R", ""))
            except Exception:
                pass

            tree = ET.ElementTree(root)
            try:
                ET.indent(tree, space="  ")    # pretty-print (Python 3.9+)
            except Exception:
                pass
            tree.write(os.path.join(day_dir, stem + ".xml"),
                       encoding="utf-8", xml_declaration=True)
        except Exception as e:
            print(f"[ERROR] | METADATA SAVE FAILED | PORT {port}: {e}")

    @staticmethod
    def rx_packet_is_valid(rx_bytes: bytes) -> bool:
        if len(rx_bytes) != 64:
            return False
        payload = rx_bytes[:60]
        rx_crc, = struct.unpack(">I", rx_bytes[60:64])
        calc_crc = zlib.crc32(payload) & 0xFFFFFFFF
        return rx_crc == calc_crc

    @staticmethod
    def tx_build_packet(payload60) -> bytes:
        payload = payload60.tobytes() if isinstance(payload60, np.ndarray) else bytes(payload60)
        if len(payload) != 60:
            raise ValueError(f"payload must be 64 bytes, got {len(payload)}")

        crc = zlib.crc32(payload) & 0xFFFFFFFF
        crc_bytes = struct.pack(">I", crc)   # ✅ unsigned
        return payload + crc_bytes

    @staticmethod
    def packet_loss(rx, loss) -> float:
        if loss == 0: return 0
        else:
            total = rx + loss
            percent_loss = (loss / total)*100
            return percent_loss


# ---- Async CSV flusher --------------------------------------------------------
# The comms thread used to call flushCsv + trimBuffers inline every 5 s. On slow
# disks a to_csv() append can hitch for tens of milliseconds, stalling the UDP
# packet loop. Offloading to a daemon thread lets the comms loop return to
# recvfrom() immediately. The queue is single-item-per-port (via de-dup) so a
# backlog can't grow indefinitely.
_flush_queue    = queue.Queue()
_flush_pending  = set()          # ports currently sitting in the queue (de-dup)
_flush_lock     = threading.Lock()
_flush_thread   = None
_flush_shutdown = False


def _flusher_worker():
    while not _flush_shutdown:
        try:
            port = _flush_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if port is None:                    # sentinel from stopFlusher
            break
        with _flush_lock:
            _flush_pending.discard(port)
        try:
            utils.flushCsv(port)
            utils.trimBuffers(port)         # legacy-list safety net (RB is a no-op)
        except Exception as e:
            print(f"[ERROR] async flush port {port}: {e}")


def submitFlush(port):
    """Schedule an async flush+trim for a device. De-duplicated — if this port
    is already in the queue, the call is dropped. Starts the worker lazily."""
    global _flush_thread
    with _flush_lock:
        if port in _flush_pending:
            return
        _flush_pending.add(port)
    _flush_queue.put(port)
    if _flush_thread is None or not _flush_thread.is_alive():
        _flush_thread = threading.Thread(target=_flusher_worker,
                                         name="csv-flusher", daemon=True)
        _flush_thread.start()


def stopFlusher(drain=True):
    """Called from App.closeEvent. Optionally drains the queue synchronously so
    the last few seconds of data actually make it to disk before the process
    exits, then signals the worker to stop."""
    global _flush_shutdown
    if drain:
        for _ in range(_flush_queue.qsize()):
            try:
                port = _flush_queue.get_nowait()
                if port is None:
                    continue
                try:    utils.flushCsv(port); utils.trimBuffers(port)
                except Exception: pass
            except queue.Empty:
                break
    _flush_shutdown = True
    _flush_queue.put(None)


class ODCont:
    """
    PI controller with integral action enabled at the first
    upward crossing of the OD setpoint.

    Error convention:
        e = target_od - current_od

    Kp < 0, Ki < 0.

    Output:
        Digital potentiometer command, 0-255 (written to bytes 24-27). This is
        a PUMP SPEED, not a dose time: the firmware must drive the dilution
        pump at this level continuously (proportional-speed mode), not treat it
        as a d_time pulse length.

    Initialisation:
        I starts at I_initial and remains constant until the
        first upward crossing of the setpoint.
        Integration is then enabled permanently.

    Logging: self.last_u (last 0-255 command; 0 while holding),
    self.last_error (last e) and self.last_I (last integral term) are read
    every sample by Comms._control_state and saved to the
    control_u / control_error / control_I CSV columns.
    """

    def __init__(
        self,
        Kp=-100.0,
        Ki=-0.02,
        update_period=1.0,
        I_initial=60.0,
        debug=False
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.dt = float(update_period)
        self.I = float(I_initial)
        self.debug = bool(debug)

        self.u_min = 0.0
        self.u_max = 255.0

        self.integral_enabled = False
        self.previous_od = None

        self.last_u = 0.0        # last 0-255 command (0 while holding)
        self.last_error = 0.0    # last e = target_od - current_od
        self.last_I = float(I_initial)  # last integral term (starts at I_initial)

        if not all(math.isfinite(x) for x in
                   [self.Kp, self.Ki, self.dt, self.I]):
            raise ValueError("Controller parameters must be finite.")

        if self.Kp >= 0 or self.Ki > 0:
            raise ValueError("Kp must be negative and Ki non-positive.")

        if self.dt <= 0:
            raise ValueError("update_period must be positive.")

    def update(self, current_od, port, target_od):

        current_od = float(current_od)
        target_od = float(target_od)

        if not (math.isfinite(current_od) and math.isfinite(target_od)):
            raise ValueError("OD measurements must be finite.")

        # Original error convention.
        e = target_od - current_od
        self.last_error = e

        # Detect first upward crossing of the setpoint.
        if not self.integral_enabled:

            crossed = (
                current_od >= target_od
                if self.previous_od is None
                else self.previous_od < target_od <= current_od
            )

            if crossed:
                self.integral_enabled = True

                if self.debug:
                    print(
                        f"[CTRL] First setpoint crossing detected. "
                        f"OD={current_od:.5f}, I={self.I:.2f}"
                    )

        # Proportional contribution.
        P = self.Kp * e

        # Integral contribution: held until first crossing.
        if self.integral_enabled:
            self.I += self.Ki * e * self.dt

        # Hold (no dilution) until the culture first reaches the setpoint.
        # Before the first upward crossing the output is forced to 0 so the
        # dilution pumps never run while the culture is still growing up to the
        # target. Otherwise the held integral bias (I_initial) leaks through and
        # raw = Kp*e + I turns the pump on while OD is still below target
        # (command > 0 whenever e < I_initial/|Kp|).
        if not self.integral_enabled:
            raw = 0.0
            command = 0
        else:
            # Total controller output.
            raw = P + self.I
            # Output saturation only (no integral anti-windup).
            command = int(round(max(self.u_min, min(self.u_max, raw))))

        voltage = 5.0 * command / 255.0

        if self.debug:
            print(
                f"[CTRL] OD={current_od:.5f}, "
                f"target={target_od:.5f}, "
                f"error={e:+.5f}"
            )

            print(
                f"[CTRL] P={P:.3f}, "
                f"I={self.I:.3f}, "
                f"raw={raw:.3f}, "
                f"command={command}/255, "
                f"voltage={voltage:.3f} V, "
                f"integral_enabled={self.integral_enabled}"
            )

        self.previous_od = current_od
        self.last_u = float(command)
        self.last_I = float(self.I)

        if command == 0:
            return self._hold(port)

        return self._pump(command, port)

    def _pump(self, pump_value, port):

        float_bytes = utils.convertFloating2Bytes(float(pump_value))

        utils.packFloatingBytes(
            fBytes=float_bytes,
            idx=24,
            port=port
        )  # Pump_value is an 8-bit digipot (0-255) pump-speed level.

        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True

        # Mark real pump activity for the despiker.
        _Vars.activeDevices[port]["_pump_last_active"] = time.time()

        # MotorState is logged centrally per-cycle in Comms.listenPorts (_pump_state).
        return pump_value, "DILUTE"

    def _hold(self, port):

        float_bytes = utils.convertFloating2Bytes(0.0)

        utils.packFloatingBytes(
            fBytes=float_bytes,
            idx=24,
            port=port
        )  # Force close.

        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True

        return 0, "HOLD"

class ODThermalCorrector:
    def __init__(self, tau_s, b0, b1, b2=0.0, T_eff0=None, T_ref=None, dt_s=1.0, use_dTdt=True) -> None:
        self.tau_s = float(tau_s)
        self.dt_s = float(dt_s)
        self.alpha = 1.0 - math.exp(-self.dt_s/self.tau_s)
        self.b0 = float(b0)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.use_dTdt = bool(use_dTdt)
        self.T_prev = None
        self.T_eff = T_eff0
        self.T_ref = T_ref
        self.OD_hat_ref = None
    
    def update(self, T, OD) -> list:
        T = float(T)
        OD = float(OD)
        if self.T_eff is None:
            self.T_eff = T
        if self.T_ref is None:
            self.T_ref = self.T_eff
        if self.T_prev is None: 
            dTdt = 0.0
        else:
            dTdt = (T - self.T_prev)  / self.dt_s
        self.T_prev = T
        self.T_eff = self.T_eff + self.alpha * (T - self.T_eff)
        OD_hat = self.b0 + self.b1 * self.T_eff
        if self.use_dTdt:
            OD_hat += self.b2 * dTdt
        if self.OD_hat_ref is None:
            self.OD_hat_ref = self.b0 + self.b1 * self.T_ref
        OD_corrected = OD - (OD_hat - self.OD_hat_ref)
        return OD_corrected, OD_hat, self.T_eff, dTdt


class TempPIController:
    """
    PI controller for temperature using relay on-time as output.
    Output is the heater on-duration in ms written to tempbbDelay (bytes 54-57).
    The microcontroller runs in bang-bang mode and uses this value as its on-time.
    """
    def __init__(self, Kp=800.0, Ki=30.0, min_ms=0, max_ms=9000, update_period=1.0, i_limit_ms=8000.0, deadband=0.1):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.min_ms = int(min_ms)
        self.max_ms = int(max_ms)
        self.dt = float(update_period)
        self.i_limit_ms = float(i_limit_ms)
        self.deadband = float(deadband)
        self.I_ms = 0.0

    def update(self, current_temp, port, target_temp):
        e = float(target_temp) - float(current_temp)
        if e <= self.deadband:
            self.I_ms = max(0.0, self.I_ms * 0.9)
            return self._set(0, port)
        P_ms = self.Kp * e
        self.I_ms += self.Ki * e * self.dt
        self.I_ms = max(0.0, min(self.i_limit_ms, self.I_ms))
        cmd_ms = int(round(max(self.min_ms, min(self.max_ms, P_ms + self.I_ms))))
        print(f"[TEMP PI] e={e:.3f}  P={P_ms:.0f}ms  I={self.I_ms:.0f}ms  cmd={cmd_ms}ms")
        return self._set(cmd_ms, port)

    def _set(self, on_ms, port):
        float_bytes = utils.convertFloating2Bytes(on_ms)
        utils.packFloatingBytes(fBytes=float_bytes, idx=14, port=port)  # heater_delay_ms at bytes 14-17
        _Vars.activeDevices[port]["DeviceVariables"]["SendMessage"] = True
        return on_ms