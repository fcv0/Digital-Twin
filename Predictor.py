import os
from collections import deque
import numpy as np
#from tensorflow.keras.models import load_model
# import joblib

# class ODMLPPredictor:
#     """
#     Live LSTM predictor that loads the model and scalers once,
#     maintains a rolling sequence buffer for time-series predictions.
#     """
#     def __init__(self, model_path: str = 'lstm_model.h5',
#                  scaler_X_path: str = 'scaler_X.pkl',
#                  scaler_y_path: str = 'scaler_y.pkl',
#                  k: float = 1.0,
#                  window_size: int = 60
#                  ):
#         os.chdir(os.path.abspath(os.path.dirname(__file__)))
#         self.model = load_model(model_path)
#         self.scaler_X = joblib.load(scaler_X_path)
#         self.scaler_y = joblib.load(scaler_y_path)
#         self.k = k
#         self.window_size = window_size

#         self.sequence_buffer = np.zeros((self.window_size, 3))

#     def predict(self, Iref: float, IrefINIT: float, IsINIT: float, Temp: float, relayState: float) -> float:
#         """
#         Runs the LSTM prediction using the current sequence buffer.
#         """
#         IrefNorm = Iref / IrefINIT
#         new_input = np.array([IrefNorm, Temp, relayState]).reshape(1, -1)
#         new_input_scaled = self.scaler_X.transform(new_input)

#         self.sequence_buffer = np.roll(self.sequence_buffer, shift=-1, axis=0)
#         self.sequence_buffer[-1, :] = new_input_scaled

#         input_sequence = self.sequence_buffer.reshape(1, self.window_size, 3)
#         y_pred_scaled = self.model.predict(input_sequence, verbose=0)
#         y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
#         y_pred = IsINIT * y_pred

#         return y_pred

#     def compute_OD(self, adj_prediction: float, main_raw_value: float) -> float:
#         """
#         Computes optical density using predicted and actual sensor values.
#         OD = 5 * log10(pred / measured)
#         """
#         return 5 * np.log10(adj_prediction / main_raw_value)

class KalmanODFilter:
    """
    Scalar Kalman filter for smoothing OD measurements.

    OD is computed as:
        z = 5 * log10(I_ref * beta / I_main)

    State variables:
      - x: filtered OD estimate
      - P: estimate covariance

    Parameters:
      Q (float): process noise variance
      R (float): measurement noise variance
      x0 (float): initial OD estimate
      P0 (float): initial estimate covariance
    """
    def __init__(self, Q: float = 1e-4, R: float = 1e-2, x0: float = 0.0, P0: float = 1.0,
                 C: float = 5.1):
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.C = C   # OD calibration constant — MUST match Utils.calcOD's
                     # CalibrationConst or KOD drifts from OD by (1 - C/Cfg) * OD.

    def update(self, I_main: float, I_ref: float, beta: float) -> float:
        """
        One Kalman update step from raw sensor readings.

        Args:
          I_main: main photodiode reading
          I_ref:  reference photodiode reading
          beta:   calibration coefficient

        Returns:
          Filtered OD estimate.
        """
        if I_main > 0 and I_ref > 0:
            z = self.C * np.log10((I_ref * beta) / I_main)
        else:
            z = self.x

        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x


class RealtimeODFilter:
    """
    Drop-in replacement for KalmanODFilter with an upstream MAD-based
    spike rejection stage.

    Pipeline per sample:
        (I_main, I_ref, beta) → OD  →  causal despiker  →  Kalman  →  float

    The interface is intentionally identical to KalmanODFilter so it can
    be swapped in Comms.py without any other changes:

        kalman = RealtimeODFilter(Q=1e-7, R=8.76e-6)

    The last spike flag is available via:

        filter.last_spike   # bool — was the most recent sample a spike?

    Design note — causal vs. centred window:
        The batch despike_series() uses center=True rolling windows, meaning
        it looks at future samples to decide whether the current point is a
        spike.  In real-time we only have past data, so the buffer is fully
        trailing.  This introduces a detection lag of roughly window//2
        samples relative to the offline version.  Everything else — the MAD
        formula, the 1.4826 scale factor, and the Kalman equations — is
        mathematically identical.

    Parameters:
      Q (float):               Kalman process noise variance.
      R (float):               Kalman measurement noise variance.
      x0 (float):              Initial Kalman OD estimate.
      P0 (float):              Initial Kalman covariance.
      window (int):            Trailing buffer length for MAD despiker (odd, ≥5).
      threshold_sigma (float): Spike rejection threshold in robust-sigma units.
    """

    def __init__(
        self,
        Q: float = 1e-4,
        R: float = 1e-2,
        x0: float = 0.0,
        P0: float = 1.0,
        window: int = 31,
        threshold_sigma: float = 5.0,
        C: float = 5.1,
        max_reject_run: int = 10,
        sigma_floor: float = 1e-4,
    ):
        if window < 5:
            raise ValueError("window must be >= 5")
        if window % 2 == 0:
            window += 1

        # Kalman state
        self.C = C
        self.Q = Q
        self.R = R
        self._x: float = x0
        self._P: float = P0
        self._P0: float = P0

        # Despiker state
        self.window = window
        self.threshold_sigma = threshold_sigma
        self._buf: deque = deque(maxlen=window)
        # Bound how long a sustained "outlier" can be rejected before we accept it
        # as a real level change (a dilution step must not be gated out forever).
        self.max_reject_run = max_reject_run
        self._reject_run = 0
        # Scale floor: stops MAD->0 on smooth stretches from flagging everything.
        self._sigma_floor = sigma_floor

        # Diagnostics
        self.last_spike: bool = False

    # ------------------------------------------------------------------
    # Public API — matches KalmanODFilter exactly
    # ------------------------------------------------------------------

    def update(self, I_main: float, I_ref: float, beta: float,
               scatter: float = None, allow_despike: bool = True) -> float:
        """
        One filter update from raw sensor readings.

        Computes OD → despiked OD → Kalman-smoothed OD.
        allow_despike=False (pump dosing) accepts the sample unconditionally —
        a dilution is a real level change and must not be gated out.
        scatter is ignored here; accepted so every filter shares one call site.

        Args:
          I_main: main photodiode reading
          I_ref:  reference photodiode reading
          beta:   calibration coefficient (e.g. np.mean(device["Coefficients"]["Beta"]))

        Returns:
          Filtered OD estimate (float), identical output type to KalmanODFilter.
        """
        # ── 1. Compute raw OD ──────────────────────────────────────────
        if I_main > 0 and I_ref > 0:
            od_raw = self.C * np.log10((I_ref * beta) / I_main)
        else:
            od_raw = self._x  # hold last estimate on bad reading

        # ── 2. Causal MAD despiker (owns buffer insertion: clean samples only) ──
        if allow_despike:
            od_clean, self.last_spike = self._despike(od_raw)
        else:
            od_clean, self.last_spike = od_raw, False
            self._buf.clear(); self._buf.append(od_raw)   # re-seed at the new level
            self._reject_run = 0

        # ── 3. Kalman update ───────────────────────────────────────────
        return self._kalman(od_clean)

    def reset(self) -> None:
        """Clear all internal state — call between experiments."""
        self._buf.clear()
        self._x  = 0.0
        self._P  = self._P0
        self._reject_run = 0
        self.last_spike = False

    @property
    def ready(self) -> bool:
        """
        True once the trailing buffer has accumulated enough samples
        for a reliable MAD estimate (~window//4 samples minimum).
        Early readings are still returned unmodified; this flag lets
        callers decide whether to trust them.
        """
        return len(self._buf) >= max(5, self.window // 4)

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _despike(self, od: float) -> tuple[float, bool]:
        """Causal MAD rejection against a CLEAN-ONLY trailing buffer.

        Returns (cleaned_od, is_spike). The sample is tested against previously
        ACCEPTED values and only enters the buffer if it passes — this method
        owns buffer insertion, update() must not pre-append.

        Why clean-only: feeding rejects back in (the old behaviour) let a
        multi-second bubble poison the very median/MAD used to reject it, so
        detection decayed sample-by-sample through the excursion.

        _reject_run bounds it: a genuine OD step (dilution) looks like a
        sustained outlier, so after max_reject_run consecutive rejections we
        accept and re-seed rather than gating it out forever.
        """
        if len(self._buf) < 3:
            self._buf.append(od)
            return od, False

        buf_arr   = np.asarray(self._buf, dtype=float)
        local_med = float(np.median(buf_arr))
        local_mad = float(np.median(np.abs(buf_arr - local_med)))
        # Floor the scale: on a smooth stretch MAD -> 0, which drives the threshold
        # to ~0 and flags every sample (the old 1e-9 floor turned this into a median
        # filter and produced most of its false positives).
        robust_sigma = max(1.4826 * local_mad, self._sigma_floor)

        is_spike = abs(od - local_med) > (self.threshold_sigma * robust_sigma)

        if is_spike:
            self._reject_run += 1
            if self._reject_run >= self.max_reject_run:
                self._buf.clear()          # sustained -> real level change
                self._buf.append(od)
                self._reject_run = 0
                return od, False
            return local_med, True

        self._reject_run = 0
        self._buf.append(od)               # clean samples only
        return od, False

    def _kalman(self, z: float) -> float:
        """Scalar random-walk Kalman update; carries state between calls."""
        P_pred  = self._P + self.Q
        K       = P_pred / (P_pred + self.R)
        self._x = self._x + K * (z - self._x)
        self._P = (1.0 - K) * P_pred
        return self._x


class RealtimeODFilterCV:
    """
    2-state *constant-velocity* Kalman filter for OD, with the same MAD
    despiker front-end and the same public interface as RealtimeODFilter
    (update / reset / .ready / .last_spike / settable .Q / .R), so it is a
    drop-in replacement.

    Why a 2-state model:
        The scalar random-walk filter has no notion of trend, so on a rising
        or falling OD it always lags by ~τ samples — you can only trade that
        lag against noise rejection.  This filter tracks BOTH the OD level and
        its rate of change (velocity), so a steady ramp is followed with *zero*
        steady-state lag for any Q > 0.  Only curvature (acceleration) induces
        lag, which lets you keep Q small (strong smoothing) and still follow
        real growth/dilution transients tightly.

    Model (discrete white-noise-acceleration / "nearly constant velocity"):
        state  x   = [od, d(od)/dt]ᵀ
        F      = [[1, dt],
                  [0,  1]]
        H      = [1, 0]                      (we measure OD only)
        Qm     = q · [[dt³/3, dt²/2],        (q = self.Q, accel spectral density)
                      [dt²/2, dt   ]]
        R      = self.R                      (measurement variance, scalar)

    Tuning:
        Keep self.R at the OD measurement variance.  self.Q (acceleration
        noise) sets how quickly the velocity estimate may change: larger =
        snappier on curvature but noisier; smaller = smoother.  Because the
        velocity state absorbs ramps, Q can be much smaller here than the
        random-walk Q for the same responsiveness.
    """

    def __init__(
        self,
        Q: float = 1e-6,
        R: float = 5e-4,
        x0: float = 0.0,
        P0: float = 1.0,
        dt: float = 1.0,
        window: int = 31,
        threshold_sigma: float = 5.0,
        C: float = 5.1,
        max_reject_run: int = 10,
        sigma_floor: float = 1e-4,
    ):
        if window < 5:
            raise ValueError("window must be >= 5")
        if window % 2 == 0:
            window += 1

        # Kalman state / params
        self.C = C          # OD calibration constant (match Utils.calcOD)
        self.Q = Q          # acceleration (process) noise spectral density
        self.R = R          # measurement noise variance
        self.dt = dt
        self._x0 = x0
        self._P0 = P0
        self._x = np.array([[x0], [0.0]], dtype=float)
        self._P = np.array([[P0, 0.0], [0.0, P0]], dtype=float)

        # Despiker state
        self.window = window
        self.threshold_sigma = threshold_sigma
        self._buf: deque = deque(maxlen=window)
        # Bound how long a sustained "outlier" can be rejected before we accept it
        # as a real level change (a dilution step must not be gated out forever).
        self.max_reject_run = max_reject_run
        self._reject_run = 0
        # Scale floor: stops MAD->0 on smooth stretches from flagging everything.
        self._sigma_floor = sigma_floor

        # Diagnostics
        self.last_spike: bool = False

    # ------------------------------------------------------------------
    # Public API — matches RealtimeODFilter
    # ------------------------------------------------------------------

    def update(self, I_main: float, I_ref: float, beta: float,
               scatter: float = None, allow_despike: bool = True) -> float:
        if I_main > 0 and I_ref > 0:
            od_raw = self.C * np.log10((I_ref * beta) / I_main)
        else:
            od_raw = float(self._x[0, 0])  # hold last estimate on bad reading

        if allow_despike:
            od_clean, self.last_spike = self._despike(od_raw)
        else:
            od_clean, self.last_spike = od_raw, False
            self._buf.clear(); self._buf.append(od_raw)
            self._reject_run = 0
        return self._kalman(od_clean)

    def reset(self) -> None:
        self._buf.clear()
        self._x = np.array([[0.0], [0.0]], dtype=float)
        self._P = np.array([[self._P0, 0.0], [0.0, self._P0]], dtype=float)
        self._reject_run = 0
        self.last_spike = False

    @property
    def ready(self) -> bool:
        return len(self._buf) >= max(5, self.window // 4)

    @property
    def velocity(self) -> float:
        """Current OD rate-of-change estimate (OD per dt)."""
        return float(self._x[1, 0])

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _despike(self, od: float) -> tuple[float, bool]:
        """Causal MAD rejection against a CLEAN-ONLY trailing buffer.

        The sample is tested against previously ACCEPTED values and only enters
        the buffer if it passes. Feeding rejects back in (the old behaviour) let a
        multi-second bubble poison the very median/MAD used to reject it, so
        detection decayed sample-by-sample through the excursion.

        _reject_run bounds this: a genuine OD step (dilution) looks like a
        sustained outlier, so after max_reject_run consecutive rejections we
        accept and re-seed the buffer rather than locking out forever.
        """
        if len(self._buf) < 3:
            self._buf.append(od)
            return od, False

        buf_arr   = np.asarray(self._buf, dtype=float)
        local_med = float(np.median(buf_arr))
        local_mad = float(np.median(np.abs(buf_arr - local_med)))
        # Floor the scale: on a very smooth stretch MAD -> 0, which would make the
        # threshold ~0 and flag every sample as a spike (the old 1e-9 floor turned
        # the despiker into a median filter and produced most of its false positives).
        robust_sigma = max(1.4826 * local_mad, self._sigma_floor)

        is_spike = abs(od - local_med) > (self.threshold_sigma * robust_sigma)

        if is_spike:
            self._reject_run += 1
            if self._reject_run >= self.max_reject_run:
                # Sustained -> treat as a real level change, not a bubble.
                self._buf.clear()
                self._buf.append(od)
                self._reject_run = 0
                return od, False
            return local_med, True

        self._reject_run = 0
        self._buf.append(od)          # clean samples only
        return od, False

    def _kalman(self, z: float) -> float:
        dt = self.dt
        F  = np.array([[1.0, dt], [0.0, 1.0]])
        H  = np.array([[1.0, 0.0]])
        q  = self.Q
        Qm = q * np.array([[dt**3 / 3.0, dt**2 / 2.0],
                           [dt**2 / 2.0, dt]])
        # Predict
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Qm
        # Update (scalar measurement) — .item() avoids numpy 2.x (1,1)->float errors
        S = (H @ P_pred @ H.T).item() + self.R
        K = (P_pred @ H.T) / S            # 2x1 gain
        y = z - (H @ x_pred).item()       # innovation
        self._x = x_pred + K * y
        self._P = (np.eye(2) - K @ H) @ P_pred
        return float(self._x[0, 0])


class RobustODFilter:
    """Constant-velocity Kalman with a bubble-aware robust front-end.

    Built against the digital twin's two-regime disturbance model, which exposes
    three specific ways the older MAD+Kalman pipeline fails:

    1. MICRO bubbles (stirrer-driven, high-rate, low-amplitude) alias into extra
       white measurement noise. The old filter had a hand-tuned fixed R, so when
       the stirrer sped up (or the reactor started churning) R was suddenly wrong
       and the filter either lagged or got jumpy.
       -> Fixed here by ADAPTIVE R, estimated from a robust scale of accepted
          innovations. R now tracks the real noise floor as the stirrer changes.

    2. LARGE bubbles (circulation-driven slugs) last several seconds — a coherent
       multi-sample excursion, not a single-point spike. The old despiker
       appended the RAW value to its own MAD buffer *before* testing it, so a
       sustained excursion progressively poisoned the very median/MAD used to
       reject it, and detection decayed sample by sample.
       -> Fixed here by keeping the robust buffer CLEAN-ONLY: rejected samples
          never enter it, so a slug can't raise its own threshold.

    3. Rejection substituted the local median — injecting a fabricated value and
       biasing the state toward stale data.
       -> Fixed here by COASTING: a rejected sample produces a predict-only step
          (no measurement update), which is the correct Bayesian handling of a
          corrupt observation. P grows while coasting, so the gate naturally
          widens and the filter re-acquires rather than locking out.

    4. A genuine step (dilution event) looks like a sustained outlier and could be
       rejected forever.
       -> Bounded by max_reject_run: after that many consecutive rejects we
          force-accept and inflate P to re-acquire.

    Interface matches RealtimeODFilter (update/reset/.ready/.last_spike/.Q/.R).
    """

    def __init__(
        self,
        Q: float = 1e-6,
        R: float = 5e-4,
        x0: float = 0.0,
        P0: float = 1.0,
        dt: float = 1.0,
        window: int = 31,
        threshold_sigma: float = 4.0,
        C: float = 5.1,
        adapt_R: bool = True,
        R_min: float = 1e-8,
        R_max: float = 1.0,
        max_reject_run: int = 12,
        warmup: int = 8,
    ):
        self.C = C                       # OD calibration constant (match Utils.calcOD)
        self.Q = Q                       # accel spectral density (process noise)
        self.R = R                       # measurement variance (adapted if adapt_R)
        self.dt = dt
        self._P0 = P0
        self._x = np.array([[x0], [0.0]], dtype=float)
        self._P = np.array([[P0, 0.0], [0.0, P0]], dtype=float)

        self.window = max(7, int(window))
        self.threshold_sigma = float(threshold_sigma)
        self.adapt_R = bool(adapt_R)
        self.R_min, self.R_max = float(R_min), float(R_max)
        self.max_reject_run = int(max_reject_run)
        self.warmup = int(warmup)

        # Innovations of ACCEPTED samples only — the robust noise-scale estimate.
        self._innov: deque = deque(maxlen=self.window)
        self._n_seen = 0
        self._reject_run = 0

        # Diagnostics
        self.last_spike: bool = False
        self.n_rejected: int = 0
        self.n_forced: int = 0

    # ---- public API ----
    def update(self, I_main: float, I_ref: float, beta: float,
               scatter: float = None, allow_despike: bool = True) -> float:
        if I_main > 0 and I_ref > 0 and beta > 0:
            z = self.C * np.log10((I_ref * beta) / I_main)
        else:
            z = float(self._x[0, 0])
        if not allow_despike:
            self._reject_run = self.max_reject_run   # force-accept this sample
        return self.update_od(z)

    def update_od(self, z: float) -> float:
        """Filter one OD measurement directly (used by benchmarks/tests)."""
        dt = self.dt
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Qm = self.Q * np.array([[dt**3 / 3.0, dt**2 / 2.0],
                                [dt**2 / 2.0, dt]])
        # --- predict ---
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Qm

        y = float(z - (H @ x_pred).item())          # innovation
        S = float((H @ P_pred @ H.T).item() + self.R)
        self._n_seen += 1

        # --- robust gate ---
        # Scale from accepted innovations (MAD), floored by the filter's own
        # predicted innovation sd so we never gate tighter than the model believes.
        sigma = max(self._robust_sigma(), np.sqrt(max(S, 1e-18)), 1e-9)
        reject = (self._n_seen > self.warmup) and (abs(y) > self.threshold_sigma * sigma)

        if reject and self._reject_run >= self.max_reject_run:
            # Sustained "outlier" = probably a real step, not a bubble. Accept and
            # inflate P so the filter snaps to the new level instead of locking out.
            reject = False
            self._P = self._P + np.eye(2) * abs(y)
            P_pred = F @ self._P @ F.T + Qm
            S = float((H @ P_pred @ H.T).item() + self.R)
            self._reject_run = 0
            self.n_forced += 1

        if reject:
            # --- coast: predict-only, no measurement update ---
            self._x, self._P = x_pred, P_pred
            self._reject_run += 1
            self.n_rejected += 1
            self.last_spike = True
            return float(self._x[0, 0])

        # --- accept: standard CV update ---
        K = (P_pred @ H.T) / S
        self._x = x_pred + K * y
        self._P = (np.eye(2) - K @ H) @ P_pred
        self._reject_run = 0
        self.last_spike = False

        self._innov.append(y)
        if self.adapt_R:
            self._adapt_R()
        return float(self._x[0, 0])

    def reset(self) -> None:
        self._x = np.array([[0.0], [0.0]], dtype=float)
        self._P = np.array([[self._P0, 0.0], [0.0, self._P0]], dtype=float)
        self._innov.clear()
        self._n_seen = 0
        self._reject_run = 0
        self.last_spike = False
        self.n_rejected = self.n_forced = 0

    @property
    def ready(self) -> bool:
        return self._n_seen > self.warmup

    @property
    def velocity(self) -> float:
        return float(self._x[1, 0])

    # ---- internals ----
    def _robust_sigma(self) -> float:
        """MAD-based sd of accepted innovations. Median-centred so a few leaked
        outliers can't drag it; 1.4826 makes it a Gaussian-consistent sd."""
        if len(self._innov) < 5:
            return 0.0
        a = np.asarray(self._innov, dtype=float)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        return 1.4826 * mad

    def _adapt_R(self) -> None:
        """Track the live measurement-noise floor (micro-bubble aliasing + detector
        noise) from accepted innovations. innovation variance ~= HP_predHᵀ + R, so
        R ~= var(innov) - P[0,0]; clamped to stay positive and bounded."""
        if len(self._innov) < max(8, self.window // 3):
            return
        s = self._robust_sigma()
        if s <= 0.0:
            return
        r_est = s * s - float(self._P[0, 0])
        r_est = min(max(r_est, self.R_min), self.R_max)
        self.R = 0.9 * self.R + 0.1 * r_est      # slew-limited so it can't chatter


class _LogSignalFilter:
    """Despike + Kalman a positive signal in LOG space.

    Bubbles perturb a photodiode multiplicatively (signal * (1+delta)), so log
    makes them additive and the filter scale-free — the same instance works on a
    ~6e6 main reading or a ~0.75 ratio without retuning.

    Same MAD/clean-buffer/Kalman math as RealtimeODFilter; this one just accepts
    an arbitrary signal instead of reconstructing OD itself.
    """

    def __init__(self, Q=1e-5, R=5e-4, window=31, threshold_sigma=5.0,
                 sigma_floor=1e-4, despike=True, smooth=True, max_reject_run=10):
        self.Q, self.R = Q, R
        self.threshold_sigma = threshold_sigma
        self.sigma_floor = sigma_floor
        self.despike, self.smooth = despike, smooth
        self.max_reject_run = max_reject_run
        self._buf = deque(maxlen=max(7, int(window)))
        self._run = 0
        self._x = None
        self._P = 1.0
        self.last_spike = False

    def update(self, v: float) -> float:
        z = float(np.log10(max(float(v), 1e-12)))
        self.last_spike = False
        if self.despike:
            if len(self._buf) < 3:
                self._buf.append(z)
            else:
                a = np.asarray(self._buf, dtype=float)
                med = float(np.median(a))
                mad = float(np.median(np.abs(a - med)))
                sig = max(1.4826 * mad, self.sigma_floor)
                if abs(z - med) > self.threshold_sigma * sig:
                    self._run += 1
                    if self._run >= self.max_reject_run:
                        self._buf.clear(); self._buf.append(z); self._run = 0
                    else:
                        z = med
                        self.last_spike = True
                else:
                    self._run = 0
                    self._buf.append(z)          # clean samples only
        if self._x is None:
            self._x = z
        if self.smooth:
            P_pred = self._P + self.Q
            K = P_pred / (P_pred + self.R)
            self._x = self._x + K * (z - self._x)
            self._P = (1.0 - K) * P_pred
        else:
            self._x = z
        return 10.0 ** self._x

    def reset(self):
        self._buf.clear(); self._run = 0; self._x = None; self._P = 1.0


class _ChannelGate:
    """Per-channel log-space MAD outlier gate. Judges ONE photodiode on its own.

    Returns (is_outlier, signed_deviation_in_sigmas). Log space because a bubble
    perturbs a PD multiplicatively, so log makes it additive and the same gate
    works on a ~6e6 main reading or a ~5e3 scatter reading without retuning.

    Clean-only buffer: rejected samples never enter, so a multi-second excursion
    can't inflate the threshold meant to reject it.
    """

    def __init__(self, window=31, threshold_sigma=5.0, sigma_floor=1e-4,
                 max_reject_run=10):
        self._buf = deque(maxlen=max(7, int(window)))
        self.threshold_sigma = threshold_sigma
        self.sigma_floor = sigma_floor
        self.max_reject_run = max_reject_run
        self._run = 0

    def update(self, v: float, allow_despike: bool = True):
        z = float(np.log10(max(float(v), 1e-12)))
        if not allow_despike:
            self._buf.clear(); self._buf.append(z); self._run = 0
            return False, 0.0, z
        if len(self._buf) < 5:
            self._buf.append(z)
            return False, 0.0, z
        a = np.asarray(self._buf, dtype=float)
        med = float(np.median(a))
        sig = max(1.4826 * float(np.median(np.abs(a - med))), self.sigma_floor)
        dev = (z - med) / sig                      # signed, in robust sigmas
        if abs(dev) > self.threshold_sigma:
            self._run += 1
            if self._run >= self.max_reject_run:   # sustained -> real level change
                self._buf.clear(); self._buf.append(z); self._run = 0
                return False, dev, med
            return True, dev, med
        self._run = 0
        self._buf.append(z)
        return False, dev, med

    def reset(self):
        self._buf.clear(); self._run = 0


class CoupledODDespiker:
    """Despike MAIN and SCATTER independently, then couple the verdicts.

    Why independent-first: a bubble can settle ON the main PD and be completely
    invisible to scatter (and vice versa). A joint//correlated test would miss
    exactly those. So each channel is judged on its own evidence, and agreement
    becomes CORROBORATION rather than a precondition.

    Why couple at all — this is the part a main-only despiker cannot do:
        a shared-path bubble    -> main UP,  scatter UP    (both perturbed, same sign)
        a real dilution         -> main UP,  scatter DOWN  (cells removed:
                                   transmission rises, so scattered light falls)
        a main-only bubble      -> main UP,  scatter flat
    Without scatter, a bubble and a dilution BOTH just read as "main jumped up",
    and a level-only despiker must either reject both or accept both. The scatter
    SIGN separates them, so we can reject the bubble and keep the dilution.

    REF is never gated: no bubble can reach it; its wiggle is source fluctuation,
    which is real signal.
    """

    def __init__(self, C: float = 5.1, window: int = 31, threshold_sigma: float = 5.0,
                 sigma_floor: float = 1e-4, max_reject_run: int = 10):
        self.C = C
        self._m = _ChannelGate(window, threshold_sigma, sigma_floor, max_reject_run)
        self._s = _ChannelGate(window, threshold_sigma, sigma_floor, max_reject_run)
        self.last_verdict = "ok"
        self.last_spike = False
        self.counts = {"ok": 0, "bubble_shared": 0, "bubble_main_only": 0,
                       "scatter_only": 0, "real_change": 0}

    def update(self, main: float, scatter: float, allow_despike: bool = True) -> tuple:
        """Returns (main_clean, verdict). main_clean is the bubble-free main
        estimate to feed the OD equation.
        allow_despike=False (pump dosing) passes everything through: a dilution
        is a real change and gating it would fight the controller."""
        m_out, m_dev, m_med = self._m.update(main, allow_despike)
        s_out, s_dev, _s_med = self._s.update(scatter, allow_despike)
        if not allow_despike:
            self.last_verdict = "pump_dosing"; self.last_spike = False
            self.counts["pump_dosing"] = self.counts.get("pump_dosing", 0) + 1
            return main, "pump_dosing" 

        verdict = "ok"
        clean = main
        if m_out and s_out:
            if m_dev * s_dev > 0:
                # both channels moved the SAME way -> nothing physical does that
                # except a disturbance in the shared path.
                verdict = "bubble_shared"
                clean = 10.0 ** m_med
            else:
                # opposite signs = the transmission/scatter relation held ->
                # this is a REAL optical change (e.g. dilution). Keep it.
                verdict = "real_change"
        elif m_out:
            # main alone is suspect: could be a bubble on the main PD, or a real
            # change scatter is too insensitive to register. Trust main's own gate.
            verdict = "bubble_main_only"
            clean = 10.0 ** m_med
        elif s_out:
            # scatter-path bubble: main is unaffected, so OD is fine. Nothing to do.
            verdict = "scatter_only"

        self.last_verdict = verdict
        self.last_spike = verdict in ("bubble_shared", "bubble_main_only")
        self.counts[verdict] = self.counts.get(verdict, 0) + 1
        return clean, verdict

    def od(self, main: float, scatter: float, ref: float, beta: float) -> float:
        clean, _ = self.update(main, scatter)
        if not (clean > 0 and ref > 0 and beta > 0):
            return 0.0
        return float(self.C * np.log10((ref * beta) / clean))

    def reset(self):
        self._m.reset(); self._s.reset()
        self.last_verdict = "ok"; self.last_spike = False
        for k in self.counts:
            self.counts[k] = 0


class ODPipelineCompare:
    """Runs three OD reconstruction pipelines side by side — a simulator-only
    diagnostic for deciding WHERE to filter.

      A : filter the OD ratio (main/ref)   — the production pipeline
      C : filter MAIN only, REF left raw   — bubbles live only in main/scatter,
                                             but ref's own detector noise then
                                             passes into OD unattenuated
      D : filter MAIN and REF              — despikes main only (ref has no
                                             bubbles) but smooths both

    All three despike main-side only; none despikes ref, because the reference
    path never sees bubbles — only source (LED) fluctuation, which is real signal.
    """

    def __init__(self, C: float = 5.1, Q: float = 1e-5, R: float = 5e-4,
                 window: int = 31, threshold_sigma: float = 5.0):
        self.C = C
        mk = lambda **kw: _LogSignalFilter(Q=Q, R=R, window=window,
                                           threshold_sigma=threshold_sigma, **kw)
        self._A  = mk()                      # on the ratio
        self._C  = mk()                      # on main
        self._Dm = mk()                      # on main
        self._Dr = mk(despike=False)         # on ref: smooth only, never despike
        # E: independent main/scatter gates, then couple (uses the scatter SIGN to
        # tell a shared bubble from a real dilution). Smoothing after the gate.
        self._E  = CoupledODDespiker(C=C, window=window, threshold_sigma=threshold_sigma)
        self._Es = _LogSignalFilter(Q=Q, R=R, window=window, despike=False)   # smooth main
        # REF must be smoothed too, or its own detector noise (~4x the LED
        # fluctuation it exists to cancel) sets the floor and hides every
        # main-side gain — that is exactly why trace C plateaus.
        self._Er = _LogSignalFilter(Q=Q, R=R, window=window, despike=False)

    def update(self, main: float, ref: float, beta: float,
               scatter: float = None) -> dict:
        if not (main > 0 and ref > 0 and beta > 0):
            return {"A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
        od = lambda m, r: self.C * np.log10((r * beta) / m)
        out = {
            "A": float(self.C * np.log10(beta / self._A.update(main / ref))),
            "C": float(od(self._C.update(main), ref)),
            "D": float(od(self._Dm.update(main), self._Dr.update(ref))),
        }
        if scatter is not None and scatter > 0:
            clean, _v = self._E.update(main, scatter)
            out["E"] = float(od(self._Es.update(clean), self._Er.update(ref)))
        else:
            out["E"] = out["D"]           # no scatter channel -> fall back
        return out

    def reset(self):
        for f in (self._A, self._C, self._Dm, self._Dr, self._E, self._Es, self._Er):
            f.reset()


class CoupledODFilter:
    """Production filter combining every change that measurably improved output.

    Pipeline (order matters — each step was validated against the twin's truth):
        main ─┬─> coupled despike (uses SCATTER sign) ─> smooth ─┐
      scatter ┘                                                   ├─> OD ─> out
        ref  ────────────────> smooth (never despiked) ──────────┘

    Why each piece:
      * despike BEFORE smoothing — smoothing a bubble first smears it over ~16
        samples at ~60% amplitude, destroying the sharpness the detector needs.
      * RAW input, no 10-sample box average — the box makes measurement noise
        ~0.9 autocorrelated, violating the Kalman's white-noise assumption
        (innovation ac1 +0.814 vs -0.08 on raw) and rendering P meaningless.
      * REF is smoothed but NEVER despiked — no bubble can reach the reference
        path, so its wiggle is real source fluctuation; but it is also an
        independent noisy PD whose noise is ~4x the LED fluctuation it cancels,
        and leaving it raw pins the error floor no matter how good main gets.
      * SCATTER disambiguates bubble from dilution: both raise main, only a
        dilution lowers scatter. Without it a despiker must reject real dilutions.
      * allow_despike=False while the dilution pump doses — a commanded OD drop
        is real and must never be gated out.

    Interface matches the other filters: update(main, ref, beta, scatter, allow_despike).
    """

    def __init__(self, Q: float = 1e-5, R: float = 5e-4, x0: float = 0.0, P0: float = 1.0,
                 dt: float = 1.0, window: int = 31, threshold_sigma: float = 5.0,
                 C: float = 5.1, sigma_floor: float = 1e-4, max_reject_run: int = 10,
                 **_ignored):
        self.C = C
        self.Q, self.R = Q, R
        self._gate = CoupledODDespiker(C=C, window=window, threshold_sigma=threshold_sigma,
                                       sigma_floor=sigma_floor, max_reject_run=max_reject_run)
        self._sm_main = _LogSignalFilter(Q=Q, R=R, window=window, despike=False)
        self._sm_ref  = _LogSignalFilter(Q=Q, R=R, window=window, despike=False)
        self.last_spike = False
        self.last_verdict = "ok"

    def update(self, I_main: float, I_ref: float, beta: float,
               scatter: float = None, allow_despike: bool = True) -> float:
        if not (I_main > 0 and I_ref > 0 and beta > 0):
            return 0.0
        if scatter is not None and scatter > 0:
            clean, verdict = self._gate.update(I_main, scatter, allow_despike)
        else:
            # No scatter channel: fall back to main's own gate (can't tell a
            # bubble from a dilution, so lean on the pump gate instead).
            clean, verdict = (I_main, "no_scatter")
        self.last_verdict = verdict
        self.last_spike = self._gate.last_spike if scatter else False
        m = self._sm_main.update(clean)
        r = self._sm_ref.update(I_ref)
        return float(self.C * np.log10((r * beta) / m))

    def reset(self):
        self._gate.reset(); self._sm_main.reset(); self._sm_ref.reset()
        self.last_spike = False; self.last_verdict = "ok"

    @property
    def counts(self):
        return self._gate.counts


def make_od_filter(model: str = "scalar", **kwargs):
    """Factory: return the requested OD filter, sharing a common interface.

    model:
      "scalar" / "rw"  -> RealtimeODFilter  (random-walk, default)
      "cv" / "2state"  -> RealtimeODFilterCV (constant-velocity, ramp-tracking)
      "robust"         -> RobustODFilter (CV + clean-only gate, coasting, adaptive R)
      "coupled"        -> CoupledODFilter (main/scatter coupled gate + ref smoothing;
                          needs the scatter channel and the pump gate)

    Unknown kwargs that one class doesn't accept (e.g. dt for the scalar
    filter) are dropped so the same call site works for both.
    """
    m = str(model).strip().lower()
    if m in ("coupled", "scatter"):
        return CoupledODFilter(**kwargs)
    if m in ("robust", "bubble", "adaptive"):
        return RobustODFilter(**kwargs)
    if m in ("cv", "2state", "2-state", "constant_velocity", "velocity"):
        return RealtimeODFilterCV(**kwargs)
    kwargs.pop("dt", None)   # scalar filter has no dt
    return RealtimeODFilter(**kwargs)