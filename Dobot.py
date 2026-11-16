"""Dobot Magician control for automated 24-well-plate sampling.

WHY THIS LIVES OUTSIDE THE UDP PROTOCOL
---------------------------------------
The bioreactor MCU (main.c) is reached over UDP; the Dobot Magician is a
completely separate device that enumerates as its own USB serial port
(Silicon Labs CP210x). It is driven directly from the host with absolute
Cartesian moves, so it consumes ZERO bytes of the 64-byte reactor packet in
byte_map.txt. The only future overlap is a stepper *sample pump*: if that pump
is wired to the ESP32 it would use the reserved bytes 54–55 (see byte_map.txt),
but the arm's XYZ motion is entirely host-side and independent.

MOBILITY MODEL
--------------
A 24-well plate is a 4-row x 6-col grid (rows A–D, cols 1–6) on ANSI/SLAS
19.3 mm pitch. Rather than teach 24 points we teach the plate's real basis:

    A1  = origin corner well          (captured pose)
    A6  = same row, +5 columns        (gives the column step vector)
    D1  = same column, +3 rows        (gives the row step vector)

    well(r, c) = A1 + c*col_step + r*row_step        (r 0..3, c 0..5)

Deriving the steps from three taught poses makes the grid robust to the plate
being rotated / skewed relative to the robot base — you are not assuming the
plate is axis-aligned. A simpler axis-aligned mode (teach A1 + a scalar pitch)
is also supported for quick setups.

HARDWARE-FREE TESTING
---------------------
If `pydobot` is not installed or no arm is connected, the controller drops into
SIMULATION mode: it tracks a virtual pose so the whole well-grid / sequencing
layer is fully exercisable before the arm arrives. Every public method works
identically in both modes; only the transport differs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ANSI/SLAS 1-2004 standard well pitch for a 24-well plate (mm).
DEFAULT_PITCH_MM = 19.3
ROWS = 4          # A, B, C, D
COLS = 6          # 1 .. 6
ROW_LABELS = "ABCD"


def well_name(row: int, col: int) -> str:
    """(0-indexed row, 0-indexed col) -> 'A1' .. 'D6'."""
    return f"{ROW_LABELS[row]}{col + 1}"


def parse_well(name: str) -> Tuple[int, int]:
    """'B3' -> (row=1, col=2). Case-insensitive."""
    name = name.strip().upper()
    row = ROW_LABELS.index(name[0])
    col = int(name[1:]) - 1
    if not (0 <= row < ROWS and 0 <= col < COLS):
        raise ValueError(f"Well {name!r} outside a 24-well plate")
    return row, col


@dataclass
class Pose:
    """Cartesian pose in the Dobot base frame. r = end-effector rotation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.r)


@dataclass
class PlateCalibration:
    """Maps a 24-well address to an absolute needle-tip target.

    Calibrated by TEACHING well centres with the needle tip itself: jog until
    the needle is centred over a well at the film surface, then capture the
    pose. Because the tool rotation R is held constant across every well, the
    fixed needle-tip<->flange offset is identical at each well and cancels — so
    teaching in 'needle space' needs no explicit tool-length measurement for XY.

    Teach >=3 corners (ideally all four: A1, A6, D1, D6). An affine map
    (translation + rotation + per-axis pitch + skew) is least-squares fitted to
    the taught points for XY; a 4th point makes it overdetermined, so random
    teaching error is averaged out and surfaced as `residual_rms/max`.

    Z is NOT taken from calibration in the current control model — it is set by
    the controller's two explicit setpoints, return_z (j) and pierce_z (k). The
    taught-Z plane fit and `penetration_depth`/`safe_clearance` below are legacy
    of the old hover+depth model and are no longer applied to motion.

    With <3 wells taught it falls back to a single taught origin + nominal
    `pitch` (axis-aligned) so basic motion still works before full calibration.
    """
    taught: dict = field(default_factory=dict)   # 'A1' -> Pose at hover point
    pitch: float = DEFAULT_PITCH_MM              # fallback when <3 taught
    safe_clearance: float = 25.0                 # mm ABOVE the plate for travel
    safe_z: float = 40.0                         # fallback travel Z (no taught pts)
    penetration_depth: float = 0.0               # mm the needle goes BELOW hover
    r_fixed: Optional[float] = None              # tool rotation, held constant

    _A: Optional[list] = field(default=None, init=False, repr=False)
    _t: Optional[list] = field(default=None, init=False, repr=False)
    _plane: Optional[list] = field(default=None, init=False, repr=False)
    residual_rms: float = field(default=0.0, init=False)
    residual_max: float = field(default=0.0, init=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def teach_well(self, name: str, pose: Pose):
        """Store a taught well centre. First taught well pins the fixed R."""
        name = name.strip().upper()
        parse_well(name)                         # validates A1..D6
        if self.r_fixed is None:
            self.r_fixed = pose.r
        self.taught[name] = pose
        self._fitted = False                     # force refit next use

    def n_taught(self) -> int:
        return len(self.taught)

    def travel_z(self) -> float:
        """Safe traverse height: ABOVE the highest taught point by safe_clearance.

        Tying travel to the taught plate (rather than a fixed absolute Z) is what
        keeps the tool from dropping BELOW the plate between wells and dragging
        through the film. Falls back to safe_z only when nothing is taught yet."""
        if self.taught:
            return max(p.z for p in self.taught.values()) + self.safe_clearance
        return self.safe_z

    def is_ready(self) -> bool:
        return len(self.taught) >= 1

    def fit(self):
        """(Re)compute the affine XY map + Z plane from taught wells."""
        import numpy as np
        if not self.taught:
            raise RuntimeError("No wells taught yet.")
        design, xs, ys, zs = [], [], [], []
        for name, p in self.taught.items():
            r, c = parse_well(name)
            design.append((c, r, 1.0))           # [col, row, 1]
            xs.append(p.x); ys.append(p.y); zs.append(p.z)

        if len(self.taught) >= 3:
            D = np.array(design)
            ax = np.linalg.lstsq(D, np.array(xs), rcond=None)[0]
            ay = np.linalg.lstsq(D, np.array(ys), rcond=None)[0]
            az = np.linalg.lstsq(D, np.array(zs), rcond=None)[0]
            self._A = [[ax[0], ax[1]], [ay[0], ay[1]]]
            self._t = [ax[2], ay[2]]
            self._plane = [az[0], az[1], az[2]]
            errs = [((ax[0]*c + ax[1]*r + ax[2] - x) ** 2 +
                     (ay[0]*c + ay[1]*r + ay[2] - y) ** 2) ** 0.5
                    for (c, r, _), x, y in zip(design, xs, ys)]
            errs = np.array(errs)
            self.residual_rms = float(np.sqrt(np.mean(errs ** 2)))
            self.residual_max = float(errs.max())
        else:
            # <3 taught: single origin + nominal axis-aligned pitch.
            if "A1" in self.taught:
                name0, p0 = "A1", self.taught["A1"]
            else:
                name0, p0 = next(iter(self.taught.items()))
            r0, c0 = parse_well(name0)
            self._A = [[self.pitch, 0.0], [0.0, self.pitch]]
            self._t = [p0.x - self.pitch * c0, p0.y - self.pitch * r0]
            self._plane = [0.0, 0.0, p0.z]
            self.residual_rms = self.residual_max = 0.0
        self._fitted = True

    def well_pose(self, row: int, col: int) -> Pose:
        """Absolute needle-tip pose at the FILM SURFACE of the given well."""
        if not self.is_ready():
            raise RuntimeError("Plate not calibrated: teach at least A1 first.")
        if not self._fitted:
            self.fit()
        x = self._A[0][0]*col + self._A[0][1]*row + self._t[0]
        y = self._A[1][0]*col + self._A[1][1]*row + self._t[1]
        z = self._plane[0]*col + self._plane[1]*row + self._plane[2]
        r = self.r_fixed if self.r_fixed is not None else 0.0
        # Cast off any numpy scalar so downstream struct.pack / config are clean.
        return Pose(float(x), float(y), float(z), float(r))


class DobotController:
    """Thin, safe wrapper over pydobot with a simulation fallback.

    Public surface used by the GUI / scripts:
        connect(port=None) / disconnect()
        home()
        pose() -> Pose
        move_to(x, y, z, r=None, wait=True)
        jog(dx=0, dy=0, dz=0, dr=0)
        set_speed(velocity, acceleration)
        calibrate            (PlateCalibration instance)
        teach(well)          capture current pose as a well centre ('A1'..'D6')
        goto_well(name)      position XY over a well at return_z (j)
        pierce() / reset_z() drive Z to pierce_z (k) / return_z (j) at current XY
        sample_all(order='snake', dwell_s=1.0, pierce=True, on_well=None)
    """

    def __init__(self):
        self._dev = None
        self.connected = False
        self.simulated = False
        self.port: Optional[str] = None
        self._sim_pose = Pose(200.0, 0.0, 50.0, 0.0)  # plausible resting pose
        self.calibrate = PlateCalibration()

        # ---- Soft workspace envelope (mm / deg) --------------------------
        # The Magician's REAL reachable set is a coupled R–Z shell, not a box;
        # this (R, Z, base-angle) envelope is a deliberately CONSERVATIVE inner
        # approximation. A commanded target outside it is trimmed to the nearest
        # in-bounds point (see clamp_target) so the arm travels to just inside
        # the boundary and stops, instead of driving into a joint limit and
        # tripping the red alarm. Widen these with set_workspace() once you've
        # mapped your own rig; taught well positions are reachable by definition
        # so they are unaffected. Units: radius/Z in mm from the J1 base axis.
        # Defaults sized to THIS arm: the identification probe found it resting
        # at R≈109 mm, Z≈-12, so r_min must sit BELOW that or real positions get
        # trimmed. These are generous starting bounds — tighten with
        # set_workspace() once you know exactly where your plate sits.
        self.r_min, self.r_max = 90.0, 310.0     # horizontal reach from base axis
        self.z_min, self.z_max = -60.0, 150.0    # vertical range
        self.angle_max_deg = 100.0               # base rotation limit (± this)
        self.clamp_enabled = True

        # ---- Alarm auto-recovery ----------------------------------------
        # If a move still trips the controller alarm (red LED), clear it and
        # carry on. Bounded so a genuine hard obstruction can't be fought
        # forever. SAFETY: auto-clearing suppresses a protective stop — keep it
        # paired with the workspace clamp and a clear physical workspace.
        self.auto_clear_alarms = True
        self._MAX_ALARM_CLEARS = 3               # per move_to call

        # ---- Custom 'soft home' reference --------------------------------
        # The Magician has absolute joint-angle ENCODERS (hand-teaching works,
        # and it reports a valid pose at power-on), so it does NOT need the flaky
        # factory homing sweep to know where it is. Our 'home' is a captured
        # reference the arm returns to. We store BOTH the Cartesian pose and the
        # raw joint angles; soft_home() prefers a JOINT-space move (MOVJ_ANGLE)
        # to home_joints — that commands each motor straight to its sensed angle,
        # so it can't be 'out of workspace' and needs no Cartesian clamp.
        self.home_pose = Pose(200.0, 0.0, 60.0, 0.0)
        self.home_joints = None                  # (j1,j2,j3,j4) deg, captured ref
        self._sim_joints = (0.0, 15.0, 15.0, 0.0)  # simulation-only joint state

        # ---- Film-piercing stroke profile --------------------------------
        # No force sensor: 'force' to pierce the film comes from a decisive,
        # high-ACCELERATION straight-down stroke plus commanding a target BELOW
        # the film so the steppers keep driving through it. The gentle approach
        # to the taught hover point uses the travel profile; only the final
        # pierce stroke uses the punchier profile. Tune per film — a stroke the
        # motors can't complete stalls and alarms (film too tough / needle bent).
        self.travel_velocity, self.travel_accel = 100.0, 100.0
        self.pierce_velocity, self.pierce_accel = 120.0, 300.0

        # ---- Explicit Z setpoints (absolute mm, Dobot base frame) ---------
        # XY comes from the plate calibration; Z is controlled directly by these
        # two values the user sets:
        #   return_z (j) = up / reset height — needle OUT; also the safe height
        #                  the arm traverses between wells at.
        #   pierce_z (k) = down height the needle drives to — pierce / inject.
        # Assumes a roughly level plate (a single absolute k for every well). The
        # taught-plane tilt compensation is no longer applied to Z in this model.
        self.return_z = 60.0     # j
        self.pierce_z = 0.0      # k

    # ---- transport -------------------------------------------------------
    # USB-serial bridges seen on Dobot Magician units. The classic Magician uses
    # a Silicon Labs CP210x; Lite / newer units use a WCH CH340/CH9102. NOTE the
    # WCH chips are ALSO common on ESP32 boards (incl. the bioreactor MCU), so a
    # chip match is a hint, not proof — when both are plugged in, pin the Dobot
    # port explicitly (DobotController.connect(port=...) / config) rather than
    # trusting auto-pick.
    _ARM_CHIP_HINTS = ("cp210", "silicon labs", "ch9102", "ch340", "wch")

    @staticmethod
    def list_ports() -> List[str]:
        """Candidate serial ports, likely-arm bridges first."""
        try:
            from serial.tools import list_ports
        except Exception:
            return []
        likely, other = [], []
        for p in list_ports.comports():
            desc = f"{p.description} {p.manufacturer or ''}".lower()
            hit = any(h in desc for h in DobotController._ARM_CHIP_HINTS)
            (likely if hit else other).append(p.device)
        return likely + other

    def connect(self, port: Optional[str] = None) -> bool:
        """Open the arm. Falls back to SIMULATION if pydobot/hardware absent.

        Returns True if a *real* arm connected, False if running simulated."""
        if port is None:
            ports = self.list_ports()
            port = ports[0] if ports else None

        try:
            from pydobot import Dobot
        except Exception:
            print("[Dobot] pydobot not installed (pip install pydobot) — SIMULATION mode.")
            self._enter_sim(port)
            return False

        if port is None:
            print("[Dobot] No serial port found — SIMULATION mode.")
            self._enter_sim(None)
            return False

        try:
            self._dev = Dobot(port=port, verbose=False)
            self.port = port
            self.connected = True
            self.simulated = False
            print(f"[Dobot] Connected on {port}.")
            return True
        except Exception as e:
            print(f"[Dobot] Connect failed on {port} ({e}) — SIMULATION mode.")
            self._enter_sim(port)
            return False

    def _enter_sim(self, port):
        self._dev = None
        self.port = port
        self.connected = True
        self.simulated = True

    def disconnect(self):
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
        self._dev = None
        self.connected = False

    # ---- motion ----------------------------------------------------------
    def pose(self) -> Pose:
        if self.simulated or self._dev is None:
            return Pose(*self._sim_pose.as_tuple())
        # pydobot's read can momentarily return None (empty serial buffer) and
        # then crash unpacking it. Guard it and fall back to the last good pose.
        try:
            x, y, z, r, *_ = self._dev.pose()
            self._last_pose = Pose(x, y, z, r)
        except (AttributeError, TypeError, IndexError, OSError):
            return getattr(self, "_last_pose", None) or Pose(200.0, 0.0, 60.0, 0.0)
        return self._last_pose

    # ---- completion wait by MOTION SETTLING -------------------------------
    # We originally waited on pydobot's queued-command index ack, but on this
    # serial link those reads return None / desync — so moves blurred together
    # and a well got visually skipped on verify / snake. Instead we watch the
    # POSE and treat a move as done once it stops changing. GetPose is a
    # self-contained request/response, so it resyncs cleanly even after a bad
    # read, and it observes physical completion directly.
    def _wait_settle(self, timeout: float = 40.0, eps: float = 0.4,
                     stable_needed: int = 3):
        """Block until the arm has physically stopped moving (or timeout)."""
        import time as _t
        _t.sleep(0.2)                       # let the commanded motion begin
        last = None
        stable = 0
        t0 = _t.time()
        while _t.time() - t0 < timeout:
            p = self.pose()
            if (last is not None and abs(p.x - last.x) < eps
                    and abs(p.y - last.y) < eps and abs(p.z - last.z) < eps):
                stable += 1
                if stable >= stable_needed:
                    return
            else:
                stable = 0
            last = p
            _t.sleep(0.09)

    # ---- workspace clamp -------------------------------------------------
    def set_workspace(self, r_min=None, r_max=None, z_min=None, z_max=None,
                      angle_max_deg=None, enabled=None):
        """Adjust the soft envelope. Any arg left None is unchanged."""
        if r_min is not None: self.r_min = float(r_min)
        if r_max is not None: self.r_max = float(r_max)
        if z_min is not None: self.z_min = float(z_min)
        if z_max is not None: self.z_max = float(z_max)
        if angle_max_deg is not None: self.angle_max_deg = float(angle_max_deg)
        if enabled is not None: self.clamp_enabled = bool(enabled)

    def clamp_target(self, x, y, z, r):
        """Trim (x,y,z) to the soft envelope. Returns (x,y,z,r,clamped,reason).

        XY is clamped in polar form (radius + base angle) so the trimmed point
        sits on the reachable boundary along the same heading — i.e. 'as far
        toward the target as we can safely go'. Z is clamped independently."""
        import math
        if not self.clamp_enabled:
            return x, y, z, r, False, ""
        reasons = []
        rad = math.hypot(x, y)
        theta = math.atan2(y, x)
        amax = math.radians(self.angle_max_deg)
        if theta > amax:
            theta = amax; reasons.append("angle+")
        elif theta < -amax:
            theta = -amax; reasons.append("angle-")
        if rad > self.r_max:
            rad = self.r_max; reasons.append("reach")
        elif rad < self.r_min:
            rad = self.r_min; reasons.append("too-close")
        nx, ny = rad * math.cos(theta), rad * math.sin(theta)
        nz = z
        if nz > self.z_max:
            nz = self.z_max; reasons.append("z-high")
        elif nz < self.z_min:
            nz = self.z_min; reasons.append("z-low")
        return nx, ny, nz, r, bool(reasons), ",".join(reasons)

    # ---- alarm handling (real hardware only) -----------------------------
    def get_alarms(self) -> bytes:
        """Raw 32-byte alarm bitmask from the controller (empty if simulated)."""
        if self.simulated or self._dev is None:
            return b""
        try:
            from pydobot.message import Message
            m = Message()
            m.id = 20          # GET_ALARMS_STATE
            m.ctrl = 0x00      # rw=0 (read), isQueued=0
            m.params = bytes([])
            resp = self._dev._send_command(m)
            return bytes(resp.params) if resp is not None else b""
        except Exception as e:
            print(f"[Dobot] get_alarms failed: {e}")
            return b""

    def is_alarmed(self) -> bool:
        """True if any alarm bit is set (red LED / lockout)."""
        return any(self.get_alarms())

    def clear_alarms(self) -> bool:
        """Clear ALL controller alarms (protocol ID 20, write). Returns ok."""
        if self.simulated or self._dev is None:
            return True
        try:
            from pydobot.message import Message
            m = Message()
            m.id = 20          # CLEAR_ALL_ALARMS_STATE (ID 20, write bit)
            m.ctrl = 0x01      # rw=1 (write), isQueued=0
            m.params = bytes([])
            self._dev._send_command(m)
            print("[Dobot] Alarms cleared.")
            return True
        except Exception as e:
            print(f"[Dobot] clear_alarms failed: {e}")
            return False

    # ---- motion ----------------------------------------------------------
    def move_to(self, x: float, y: float, z: float,
                r: Optional[float] = None, wait: bool = True,
                linear: bool = False):
        """Absolute move to (x, y, z, r), trimmed to the soft workspace.

        linear=False (default) uses MOVJ (joint interpolation): only the
        endpoints need be reachable, so it rarely trips a limit mid-path.
        linear=True uses MOVL (straight line) for clean vertical descents.

        Returns True if the target had to be clamped, else False."""
        if r is None:
            r = self.pose().r
        x, y, z, r, clamped, why = self.clamp_target(x, y, z, r)
        if clamped:
            print(f"[Dobot] target clamped ({why}) -> "
                  f"({x:.1f}, {y:.1f}, {z:.1f})")

        if self.simulated or self._dev is None:
            self._sim_pose = Pose(x, y, z, r)
            if wait:
                time.sleep(0.05)
            return clamped

        # Clear any lingering lock so this command is actually accepted.
        if self.auto_clear_alarms and self.is_alarmed():
            self.clear_alarms()

        from pydobot.enums import PTPMode
        mode = PTPMode.MOVL_XYZ if linear else PTPMode.MOVJ_XYZ
        # Send via the wait=False path (never crashes on a None read) and do our
        # own None-safe completion wait — see _wait_queue.
        self._dev._set_ptp_cmd(x, y, z, r, mode=mode, wait=False)
        if wait:
            self._wait_settle()

        # If the move itself tripped an alarm, clear it (bounded) and continue.
        if self.auto_clear_alarms:
            cleared = 0
            while self.is_alarmed() and cleared < self._MAX_ALARM_CLEARS:
                self.clear_alarms()
                cleared += 1
                time.sleep(0.05)
            if cleared and self.is_alarmed():
                print("[Dobot] alarm persists after clears — likely a real "
                      "obstruction or hard limit; check the arm.")
        return clamped

    def jog(self, dx=0.0, dy=0.0, dz=0.0, dr=0.0, wait=True):
        """Relative nudge from the current pose."""
        p = self.pose()
        self.move_to(p.x + dx, p.y + dy, p.z + dz, p.r + dr, wait=wait)

    def joints(self) -> Tuple[float, float, float, float]:
        """Absolute joint angles (j1, j2, j3, j4) in degrees from the encoders."""
        if self.simulated or self._dev is None:
            return self._sim_joints
        try:
            x, y, z, r, j1, j2, j3, j4 = self._dev.pose()
            self._last_joints = (j1, j2, j3, j4)
        except (AttributeError, TypeError, ValueError, OSError):
            return getattr(self, "_last_joints", None) or self._sim_joints
        return self._last_joints

    def joint_move(self, j1, j2, j3, j4, wait: bool = True):
        """MOVJ in JOINT/angle space to absolute joint angles (deg).

        Targets are motor angles referenced by the absolute encoders, so this
        bypasses the Cartesian workspace clamp — used for the joint-space home."""
        if self.simulated or self._dev is None:
            self._sim_joints = (j1, j2, j3, j4)
            return
        from pydobot.enums import PTPMode
        if self.auto_clear_alarms and self.is_alarmed():
            self.clear_alarms()
        self._dev._set_ptp_cmd(j1, j2, j3, j4,
                               mode=PTPMode.MOVJ_ANGLE, wait=False)
        if wait:
            self._wait_settle()

    def set_home_pose(self, pose: Optional[Pose] = None) -> Pose:
        """Define the home reference (defaults to the CURRENT pose).

        Captures BOTH the Cartesian pose and the raw joint angles, so soft_home
        can return here in joint space."""
        self.home_pose = pose or self.pose()
        try:
            self.home_joints = self.joints()
        except Exception as e:
            print(f"[Dobot] (could not read joints for home: {e})")
        p = self.home_pose
        jt = ("  joints " + ", ".join(f"{v:.1f}" for v in self.home_joints)
              if self.home_joints else "")
        print(f"[Dobot] Home reference set to ({p.x:.1f}, {p.y:.1f}, {p.z:.1f}){jt}")
        return self.home_pose

    def soft_home(self, wait: bool = True):
        """Custom 'home' — NO factory sweep. Prefers a JOINT-space move to the
        captured joint angles (each motor to its sensed angle, clamp-free); falls
        back to a clamped Cartesian MOVJ if no joint reference has been set."""
        if self.simulated or self._dev is None:
            if self.home_joints:
                self._sim_joints = self.home_joints
            self._sim_pose = Pose(*self.home_pose.as_tuple())
            return
        if self.home_joints is not None:
            self.joint_move(*self.home_joints, wait=wait)
            print("[Dobot] Soft-home (joint-space) complete "
                  f"— joints {', '.join(f'{v:.1f}' for v in self.home_joints)}.")
        else:
            p = self.home_pose
            self.move_to(p.x, p.y, p.z, p.r, wait=wait)
            print("[Dobot] Soft-home (cartesian) complete "
                  f"({p.x:.1f}, {p.y:.1f}, {p.z:.1f}). Tip: click Set Home to "
                  "capture a joint-space reference for a more robust home.")

    def set_home_params(self, x: float = 200.0, y: float = 0.0,
                        z: float = 0.0, r: float = 0.0):
        """Set the factory-home ENDPOINT (protocol ID 30, SetHOMEParams).

        This is where home()/SetHOMECmd (ID 31) drives the arm at the end of its
        referencing sweep. Sent as a queued write; the subsequent home command
        then targets this point instead of the flaky factory default."""
        if self.simulated or self._dev is None:
            self.home_pose = Pose(x, y, z, r)
            return
        import struct
        from pydobot.message import Message
        m = Message()
        m.id = 30            # SET_GET_HOME_PARAMS
        m.ctrl = 0x03        # rw=1 (write), isQueued=1
        m.params = bytearray()
        for v in (x, y, z, r):
            m.params.extend(struct.pack("f", float(v)))
        self._dev._send_command(m)
        print(f"[Dobot] HOMEParams set to ({x:.1f}, {y:.1f}, {z:.1f}, {r:.1f}).")

    def recover(self, wait: bool = True):
        """Emergency un-stick: clear alarms, then MOVJ to a known safe centre
        pose, bypassing the workspace clamp so a bad current position can't
        swing the target. Motion is gated on alarm state, so the clear must land
        first. If the arm is on a HARD mechanical limit this can still re-alarm —
        in that case hand-move it: hold the unlock button on the forearm and
        gently guide the arm back into the workspace, then Clear Alarm."""
        SAFE = Pose(200.0, 0.0, 60.0, 0.0)
        if self.simulated or self._dev is None:
            self._sim_pose = Pose(*SAFE.as_tuple())
            return
        self.clear_alarms()
        prev = self.clamp_enabled
        try:
            self.clamp_enabled = False        # go to the hardcoded-safe target as-is
            self.move_to(SAFE.x, SAFE.y, SAFE.z, SAFE.r, wait=wait)
        finally:
            self.clamp_enabled = prev
        print("[Dobot] Recover attempted -> (200, 0, 60). If still alarmed, "
              "hand-move via the arm's unlock button.")

    def home(self, wait: bool = False):
        """FACTORY homing routine via protocol SET_HOME_CMD (ID 31) — advanced.

        This is the built-in referencing sweep (slow, ~15 s, slam-prone on the
        classic Magician). Prefer soft_home() for day-to-day parking; use this
        only if you suspect the step counters have drifted from the sensors.

        This pydobot build ships no home() helper, so we send the queued home
        command directly. wait=False returns immediately; wait=True blocks until
        the queued command completes. Any lingering alarm is cleared first."""
        if self.simulated or self._dev is None:
            self._sim_pose = Pose(200.0, 0.0, 50.0, 0.0)
            return
        import struct
        from pydobot.message import Message
        if self.auto_clear_alarms and self.is_alarmed():
            self.clear_alarms()
        try:
            m = Message()
            m.id = 31            # SET_HOME_CMD
            m.ctrl = 0x03        # rw=1, isQueued=1
            m.params = bytearray(struct.pack("I", 0))   # reserved payload
            self._dev._send_command(m)                  # wait=False -> None-safe
            if wait:
                self._wait_settle(timeout=60.0)         # sweep can take a while
            print("[Dobot] Homing started." + (" (done)" if wait else ""))
        except Exception as e:
            print(f"[Dobot] Homing failed: {e}")

    def set_speed(self, velocity: float = 100.0, acceleration: float = 100.0):
        if self.simulated or self._dev is None:
            return
        try:
            self._dev.speed(velocity, acceleration)
        except Exception as e:
            print(f"[Dobot] set_speed failed: {e}")

    # ---- plate teaching / addressing ------------------------------------
    def teach(self, well: str):
        """Capture the CURRENT needle-tip pose as the centre of `well`.

        Accepts any well A1..D6. Teach the four corners (A1, A6, D1, D6) with
        the needle centred at the film surface for an overdetermined fit."""
        p = self.pose()
        self.calibrate.teach_well(well, p)
        print(f"[Dobot] Taught {well.upper()} = ({p.x:.1f}, {p.y:.1f}, "
              f"{p.z:.1f}); {self.calibrate.n_taught()} well(s) taught. "
              f"{self.calibration_report()}")
        return p

    def calibration_report(self) -> str:
        """Human-readable calibration quality (fit residual)."""
        c = self.calibrate
        n = c.n_taught()
        if n == 0:
            return "not calibrated"
        c.fit()
        if n >= 3:
            return (f"fit over {n} wells — residual rms {c.residual_rms:.2f} mm, "
                    f"max {c.residual_max:.2f} mm")
        return (f"{n} well(s) taught — single-point + {c.pitch:.1f} mm pitch "
                f"(teach ≥3 corners for a real fit)")

    def verify_corners(self, wait: bool = True):
        """Tour the four corner wells at the return height so you can eyeball
        needle centring before committing to a full run. No piercing."""
        for name in ("A1", "A6", "D6", "D1"):     # perimeter order
            self.goto_well(name, wait=wait)
            time.sleep(0.4)

    def goto_well(self, name: str, wait: bool = True):
        """Position the needle over a well's XY at the RETURN height (j).

        Z control is separate and explicit: pierce() drives down to pierce_z (k),
        reset_z() returns up to return_z (j). Motion here is lift-in-place to the
        return height, then traverse — so the needle never drags across the plate
        (as long as return_z sits above it)."""
        row, col = parse_well(name)
        target = self.calibrate.well_pose(row, col)   # XY from the affine fit
        j = self.return_z
        cur = self.pose()
        z_travel = max(cur.z, j)                       # never below return height
        self.move_to(cur.x, cur.y, z_travel, target.r, wait=wait)      # lift in place
        self.move_to(target.x, target.y, z_travel, target.r, wait=wait)  # traverse
        if z_travel > j:                               # settle down to j over the well
            self.move_to(target.x, target.y, j, target.r, wait=wait, linear=True)
        return target

    def pierce(self, wait: bool = True):
        """Drive Z straight down to pierce_z (k) at the CURRENT XY, using the
        forceful pierce profile, then restore the travel profile."""
        p = self.pose()
        if self.simulated or self._dev is None:
            self._sim_pose = Pose(p.x, p.y, self.pierce_z, p.r)
            return
        self.set_speed(self.pierce_velocity, self.pierce_accel)   # punchy profile
        try:
            self.move_to(p.x, p.y, self.pierce_z, p.r, wait=wait, linear=True)
        finally:
            self.set_speed(self.travel_velocity, self.travel_accel)
        if self.is_alarmed():
            print("[Dobot] pierce alarmed — film too tough for the arm's torque, "
                  "or the needle bottomed out. Raise pierce_z / check the needle.")

    def reset_z(self, wait: bool = True):
        """Return Z straight up to return_z (j) at the CURRENT XY (needle out)."""
        p = self.pose()
        self.set_speed(self.travel_velocity, self.travel_accel)
        self.move_to(p.x, p.y, self.return_z, p.r, wait=wait, linear=True)

    # Back-compat alias: older callers say retract(); it's the Z reset.
    def retract(self, wait: bool = True):
        self.reset_z(wait=wait)

    def sample_well(self, name: str, dwell_s: float = 0.5, wait: bool = True):
        """Full single-well cycle: XY over the well (j) -> pierce down (k) ->
        dwell (inject window) -> RETURN Z up (j). Always leaves the needle out,
        so stepping well-to-well never drags. This is the safe per-well action."""
        self.goto_well(name, wait=wait)       # XY at return height j
        self.pierce(wait=wait)                 # down to pierce height k
        if dwell_s:
            time.sleep(dwell_s)                # inject window (future pump hook)
        self.reset_z(wait=wait)                # back up to j before anything else

    def iter_wells(self, order: str = "row"):
        """Yield well names in visiting order.

        'row'    -> A1,A2..A6,B1..  (serpentine off; simple raster)
        'snake'  -> A1..A6,B6..B1   (minimises travel between rows)
        'col'    -> A1,B1,C1,D1,A2..
        """
        if order == "col":
            for c in range(COLS):
                for r in range(ROWS):
                    yield well_name(r, c)
        elif order == "snake":
            for r in range(ROWS):
                cols = range(COLS) if r % 2 == 0 else reversed(range(COLS))
                for c in cols:
                    yield well_name(r, c)
        else:  # 'row'
            for r in range(ROWS):
                for c in range(COLS):
                    yield well_name(r, c)

    def sample_all(self, order: str = "snake", dwell_s: float = 1.0,
                   pierce: bool = True, on_well=None, stop_flag=None):
        """Visit every well: XY over the well (at return_z j) -> optional pierce
        to k -> dwell (inject window) -> reset to j -> next well.

        pierce=False does an XY-only tour (dry run) at the return height.
        on_well(name)  -> fired at the bottom of the stroke (future pump trigger).
        stop_flag()    -> return True to abort cleanly.
        """
        if not self.calibrate.is_ready():
            raise RuntimeError("Calibrate the plate (teach A1) before sampling.")
        for name in self.iter_wells(order):
            if stop_flag is not None and stop_flag():
                print("[Dobot] sample_all aborted.")
                break
            self.goto_well(name)              # XY over the well at j
            if pierce:
                self.pierce()                 # down to k
            if on_well is not None:
                on_well(name)                 # hook: pump/inject at the bottom
            if dwell_s:
                time.sleep(dwell_s)
            self.reset_z()                    # up to j — needle out before next well
        print("[Dobot] Sampling sequence complete.")


# --------------------------------------------------------------------------
# Optional PyQt6 tab. Import lazily so the module stays usable head-less.
# Wire into App.py exactly like the Microscope tab, e.g.:
#     from Dobot import build_dobot_tab
#     self.tabs.addTab(build_dobot_tab(), "Sampling")
# --------------------------------------------------------------------------
def build_dobot_tab(controller: Optional[DobotController] = None):
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
        QLineEdit, QComboBox, QFrame, QCheckBox,
    )
    from PyQt6.QtCore import Qt, QTimer

    ctrl = controller or DobotController()

    ACCENT, MUTED, TEXT = "#1bb89a", "#8a93a3", "#e6e9ef"
    w = QWidget()
    root = QVBoxLayout(w)

    DANGER = "#e05a5a"
    status = QLabel("Disconnected")
    status.setStyleSheet(f"color:{MUTED}; font-weight:600;")
    alarm_lbl = QLabel("")           # shows red lockout state
    alarm_lbl.setStyleSheet(f"color:{MUTED}; font-weight:600;")
    pose_lbl = QLabel("X — Y — Z — R —")
    pose_lbl.setStyleSheet(f"color:{TEXT}; font-family:Consolas; font-size:14px;")

    _tick = [0]                      # cheap frame counter for slow alarm polls

    def refresh(check_alarm: bool = False):
        # Must never raise: this runs from a QTimer, and an unhandled exception
        # in a Qt slot aborts the whole app (PyQt6 qFatal). A serial read can
        # transiently fail, so swallow and show it instead of crashing.
        try:
            if not ctrl.connected:
                return
            mode = "SIMULATION" if ctrl.simulated else f"Connected · {ctrl.port}"
            status.setText(mode)
            status.setStyleSheet(f"color:{ACCENT}; font-weight:600;")
            p = ctrl.pose()
            pose_lbl.setText(f"X {p.x:7.1f}   Y {p.y:7.1f}   Z {p.z:7.1f}   R {p.r:6.1f}")
            # Alarm poll is a second serial round-trip, so only every ~8th tick
            # (~4 s) or when a caller explicitly asks after a move.
            if check_alarm and not ctrl.simulated:
                if ctrl.is_alarmed():
                    alarm_lbl.setText("● ALARM / LOCKOUT")
                    alarm_lbl.setStyleSheet(f"color:{DANGER}; font-weight:700;")
                else:
                    alarm_lbl.setText("● OK")
                    alarm_lbl.setStyleSheet(f"color:{ACCENT}; font-weight:600;")
        except Exception as e:
            status.setText(f"read error: {e}")
            status.setStyleSheet(f"color:{DANGER}; font-weight:600;")

    def tick():
        if not ctrl.connected:
            return
        if _homing["active"]:
            return               # hold the 'homing…' status; don't contend on serial
        _tick[0] += 1
        refresh(check_alarm=(_tick[0] % 8 == 0))

    def guard(fn):
        """Wrap a button handler so a raised exception shows in the status bar
        instead of propagating out of the Qt slot — an unhandled exception in a
        PyQt6 slot aborts the whole process (the exit-127 crashes). Common
        trigger: 'Go to well' / 'Sample all' before the plate is calibrated."""
        def slot(*_):
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc()
                status.setText(f"⚠ {e}")
                status.setStyleSheet(f"color:{DANGER}; font-weight:600;")
        return slot

    def on_connect():
        ctrl.connect()
        refresh(check_alarm=True)

    def on_clear():
        ctrl.clear_alarms()
        refresh(check_alarm=True)

    # Long moves (homing/parking) are CONFIRMED complete before we trust
    # coordinates, but blocking would freeze the UI — so run them on a worker
    # thread (wait=True polls the queued-command index to completion) and flip a
    # flag the tick loop watches. pydobot's internal lock serialises the worker's
    # serial access against the pose-poll timer, so this is safe.
    _homing = {"active": False}

    def run_bg(fn, msg):
        if _homing["active"] or not ctrl.connected:
            return
        _homing["active"] = True
        status.setText(msg)
        status.setStyleSheet(f"color:{ACCENT}; font-weight:600;")

        def worker():
            try:
                fn()
            except Exception as e:
                print(f"[Dobot] background op error: {e}")
            finally:
                _homing["active"] = False       # tick() picks it up, re-refreshes
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def on_set_home():
        ctrl.set_home_pose()                    # capture current pose as reference
        status.setText(f"home ref set: "
                       f"({ctrl.home_pose.x:.0f},{ctrl.home_pose.y:.0f},{ctrl.home_pose.z:.0f})")

    def on_disconnect():
        ctrl.disconnect()                       # closes the serial port
        status.setText("Disconnected")
        status.setStyleSheet(f"color:{MUTED}; font-weight:600;")
        alarm_lbl.setText("")
        pose_lbl.setText("X — Y — Z — R —")

    # Factory-home endpoint fields (SetHOMEParams, ID 30). Factory home sets
    # these first, then runs the sweep — so the arm ends at a point you choose
    # instead of the flaky default.
    hp_x = QLineEdit("200"); hp_y = QLineEdit("0"); hp_z = QLineEdit("0")
    for f in (hp_x, hp_y, hp_z):
        f.setFixedWidth(52)

    def on_factory_home():
        x = float(hp_x.text() or 0); y = float(hp_y.text() or 0); z = float(hp_z.text() or 0)
        def work():
            ctrl.set_home_params(x, y, z, 0.0)   # ID 30: define the endpoint
            ctrl.home(wait=True)                 # ID 31: sweep to it, confirmed
        run_bg(work, f"factory homing → ({x:.0f},{y:.0f},{z:.0f})… keep clear")

    # --- connection row ---
    top = QHBoxLayout()
    b_conn = QPushButton("Connect"); b_conn.clicked.connect(guard(on_connect))
    b_disc = QPushButton("Disconnect"); b_disc.clicked.connect(guard(on_disconnect))
    b_home = QPushButton("Home")     # soft home: gentle MOVJ to the reference pose
    b_home.clicked.connect(guard(lambda: run_bg(
        lambda: ctrl.soft_home(wait=True), "moving to home ref…")))
    b_sethome = QPushButton("Set Home"); b_sethome.clicked.connect(guard(on_set_home))
    b_fhome = QPushButton("Factory home")   # sets HOMEParams then runs the sweep
    b_fhome.clicked.connect(guard(on_factory_home))
    b_clear = QPushButton("Clear Alarm"); b_clear.clicked.connect(guard(on_clear))
    b_recover = QPushButton("Recover")   # clear + go to safe centre (clamp bypassed)
    b_recover.clicked.connect(guard(lambda: run_bg(
        lambda: ctrl.recover(wait=True), "recovering → (200,0,60)…")))
    chk_auto = QCheckBox("Auto-clear")
    chk_auto.setChecked(ctrl.auto_clear_alarms)
    chk_auto.toggled.connect(lambda v: setattr(ctrl, "auto_clear_alarms", v))
    top.addWidget(b_conn); top.addWidget(b_disc); top.addWidget(b_home)
    top.addWidget(b_sethome); top.addWidget(b_fhome)
    top.addWidget(b_clear); top.addWidget(b_recover)
    top.addWidget(chk_auto); top.addWidget(status); top.addStretch()
    top.addWidget(alarm_lbl)
    root.addLayout(top)

    hp_row = QHBoxLayout()
    hp_row.addWidget(QLabel("Factory-home target  X:")); hp_row.addWidget(hp_x)
    hp_row.addWidget(QLabel("Y:")); hp_row.addWidget(hp_y)
    hp_row.addWidget(QLabel("Z:")); hp_row.addWidget(hp_z)
    hp_row.addStretch()
    root.addLayout(hp_row)
    root.addWidget(pose_lbl)

    # --- jog pad ---
    jog = QGridLayout()
    step = QLineEdit("5.0"); step.setFixedWidth(60)

    def mk_jog(dx=0, dy=0, dz=0, label=""):
        b = QPushButton(label)
        def go():
            s = float(step.text() or 0)
            ctrl.jog(dx=dx * s, dy=dy * s, dz=dz * s); refresh(check_alarm=True)
        b.clicked.connect(guard(go))
        return b

    jog.addWidget(QLabel("Step (mm):"), 0, 0); jog.addWidget(step, 0, 1)
    jog.addWidget(mk_jog(dy=+1, label="Y+"), 1, 1)
    jog.addWidget(mk_jog(dx=-1, label="X-"), 2, 0)
    jog.addWidget(mk_jog(dx=+1, label="X+"), 2, 2)
    jog.addWidget(mk_jog(dy=-1, label="Y-"), 3, 1)
    jog.addWidget(mk_jog(dz=+1, label="Z+"), 1, 3)
    jog.addWidget(mk_jog(dz=-1, label="Z-"), 3, 3)
    root.addLayout(jog)

    # --- teach the four corners (needle tip centred at the film surface) ---
    cal_lbl = QLabel("not calibrated")
    cal_lbl.setStyleSheet(f"color:{MUTED};")

    def update_cal():
        rpt = ctrl.calibration_report()
        cal_lbl.setText(rpt)
        # green once we have a real (>=3 well) fit with a tight residual
        good = ("residual" in rpt and ctrl.calibrate.residual_max <= 0.5)
        cal_lbl.setStyleSheet(f"color:{ACCENT if good else MUTED};")

    teach_row = QHBoxLayout()
    for corner in ("A1", "A6", "D1", "D6"):
        b = QPushButton(f"Teach {corner}")
        b.clicked.connect(guard(lambda c=corner: (ctrl.teach(c), update_cal(), refresh())))
        teach_row.addWidget(b)
    root.addLayout(teach_row)
    root.addWidget(cal_lbl)

    # --- Z setpoints: return_z (j, up) and pierce_z (k, down) + force profile ---
    z_row = QHBoxLayout()
    retz = QLineEdit(str(ctrl.return_z)); retz.setFixedWidth(55)
    prcz = QLineEdit(str(ctrl.pierce_z)); prcz.setFixedWidth(55)
    pv = QLineEdit(str(ctrl.pierce_velocity)); pv.setFixedWidth(44)
    pa = QLineEdit(str(ctrl.pierce_accel)); pa.setFixedWidth(44)

    def apply_z():
        try: ctrl.return_z = float(retz.text() or 0)
        except ValueError: pass
        try: ctrl.pierce_z = float(prcz.text() or 0)
        except ValueError: pass
        try: ctrl.pierce_velocity = float(pv.text() or 0)
        except ValueError: pass
        try: ctrl.pierce_accel = float(pa.text() or 0)
        except ValueError: pass
    for f in (retz, prcz, pv, pa):
        f.editingFinished.connect(apply_z)

    def set_ret_cur():
        ctrl.return_z = round(ctrl.pose().z, 1); retz.setText(str(ctrl.return_z))
    def set_prc_cur():
        ctrl.pierce_z = round(ctrl.pose().z, 1); prcz.setText(str(ctrl.pierce_z))
    b_setret = QPushButton("=Z"); b_setret.clicked.connect(guard(set_ret_cur))
    b_setprc = QPushButton("=Z"); b_setprc.clicked.connect(guard(set_prc_cur))

    z_row.addWidget(QLabel("Return Z (j):")); z_row.addWidget(retz); z_row.addWidget(b_setret)
    z_row.addWidget(QLabel("  Pierce Z (k):")); z_row.addWidget(prcz); z_row.addWidget(b_setprc)
    z_row.addWidget(QLabel("  vel:")); z_row.addWidget(pv)
    z_row.addWidget(QLabel("accel:")); z_row.addWidget(pa)
    z_row.addStretch()
    root.addLayout(z_row)

    # --- position (XY) then explicit Z control ---
    go_row = QHBoxLayout()
    well = QComboBox()
    well.addItems([well_name(r, c) for r in range(ROWS) for c in range(COLS)])
    b_go = QPushButton("Go to well")     # XY over the well at return_z (j)
    b_go.clicked.connect(guard(lambda: (ctrl.goto_well(well.currentText()), refresh(check_alarm=True))))
    b_pwell = QPushButton("Pierce well")    # full cycle: go→pierce→dwell→RESET
    b_pwell.clicked.connect(guard(lambda: (ctrl.sample_well(well.currentText()), refresh(check_alarm=True))))
    b_pierce = QPushButton("Pierce (→k)")   # manual: Z down to k, stays down
    b_pierce.clicked.connect(guard(lambda: (ctrl.pierce(), refresh(check_alarm=True))))
    b_reset = QPushButton("Reset (→j)")     # manual: Z up to j
    b_reset.clicked.connect(guard(lambda: (ctrl.reset_z(), refresh(check_alarm=True))))
    b_verify = QPushButton("Verify corners")
    b_verify.clicked.connect(guard(lambda: (ctrl.verify_corners(), refresh(check_alarm=True))))
    b_all = QPushButton("Sample all (snake)")  # goto→pierce→dwell→reset each well
    b_all.clicked.connect(guard(lambda: ctrl.sample_all(order="snake", dwell_s=0.5)))
    for wdg in (well, b_go, b_pwell, b_pierce, b_reset, b_verify, b_all):
        go_row.addWidget(wdg)
    go_row.addStretch()
    root.addLayout(go_row)
    root.addStretch()

    # Live pose readout: poll twice a second whenever connected. Cheap (one
    # serial pose read); harmless in simulation (returns the virtual pose).
    _timer = QTimer(w)
    _timer.timeout.connect(tick)
    _timer.start(500)
    w._dobot_timer = _timer          # keep a ref so it isn't GC'd
    w.controller = ctrl              # expose for external wiring
    # Stop polling the moment the widget is torn down, so a timer tick can't
    # touch the serial port mid-shutdown (that race aborts the process).
    w.destroyed.connect(_timer.stop)
    refresh()
    return w


if __name__ == "__main__":
    # Head-less test of the affine 4-corner fit (no Qt, no hardware).
    # Simulate a plate mounted rotated 7° and tilted 1 mm across, with the well
    # centres taught imperfectly (±0.15 mm eyeballing noise), then check the fit
    # recovers every well centre to well under a well radius.
    import math, random
    random.seed(1)
    pitch = 19.3
    theta = math.radians(7.0)                 # plate yaw vs robot base
    ox, oy, oz = 180.0, -40.0, 5.0            # A1 centre in robot frame
    cos, sin = math.cos(theta), math.sin(theta)

    def true_well(row, col):                  # ground-truth well centre
        x = ox + (col * pitch) * cos - (row * pitch) * sin
        y = oy + (col * pitch) * sin + (row * pitch) * cos
        z = oz + 0.03 * col - 0.02 * row      # gentle tilt
        return x, y, z

    c = DobotController()
    c.connect()                               # SIMULATION unless an arm present
    for name in ("A1", "A6", "D1", "D6"):
        r, col = parse_well(name)
        tx, ty, tz = true_well(r, col)
        noisy = Pose(tx + random.gauss(0, 0.15),
                     ty + random.gauss(0, 0.15),
                     tz + random.gauss(0, 0.05), 0.0)
        c.calibrate.teach_well(name, noisy)   # teach the (noisy) corner
    print(c.calibration_report())

    worst = 0.0
    for r in range(ROWS):
        for col in range(COLS):
            p = c.calibrate.well_pose(r, col)
            tx, ty, tz = true_well(r, col)
            err = math.hypot(p.x - tx, p.y - ty)
            worst = max(worst, err)
    print(f"Worst predicted-vs-true well-centre error over all 24: {worst:.3f} mm")
    print("A1 ->", tuple(round(v, 2) for v in c.calibrate.well_pose(0, 0).as_tuple()))
    print("D6 ->", tuple(round(v, 2) for v in c.calibrate.well_pose(3, 5).as_tuple()))
