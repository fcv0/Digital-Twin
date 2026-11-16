"""Physics-based digital twin of the cell-density sensor, adapted for real-time
use inside the GUI's offline mode.

Ported from Digital_Twin_colab_ready.ipynb — the Colab/plotting/pandas bits are
removed so the core depends only on numpy. Three layers are preserved:

  1. Forward physical model : LED, optics, thermal plant, biomass growth
  2. Sensor/disturbance model: detector noise, scatter channel, bubbles, OD recon
  3. Orchestration          : SensorTwinEngine.step() advances one dt and returns
                              one row of {main_signal, ref_signal, scatter_signal,
                              temperature_C, ...}.

Real-time API (used by Comms.simulate_twin):
    engine = build_engine(target_temp_C=37.0, seed=port, mu_mean=..., ...)
    engine.set_target_temp(t)     # live-couple the GUI temperature setpoint
    engine.set_motor_speed(s)     # live-couple aeration → bubble rate
    row = engine.step()           # one simulated second

The GUI ignores the twin's own beta/od (it re-derives them from the raw
Main/Ref signals exactly as it would for a live reactor); the twin just supplies
realistic raw optical signals + temperature.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# ============================== configuration ==============================

@dataclass
class OpticalCalibration:
    main_obs: float = 6_000_000.0
    ref_obs: float = 8_000_000.0
    scatter_obs: float = 5_000.0


@dataclass
class SimulationConfig:
    dt_s: float = 1.0
    max_steps: int = 20_000
    beta_warmup_s: float = 1200.0
    od_calibration_constant: float = 5.1
    random_seed: Optional[int] = 1


@dataclass
class LEDConfig:
    drive_current_mA: float = 50.0
    slope_eff_W_per_A: float = 0.36
    T_ref_C: float = 25.0
    temp_coeff_per_C: float = -0.003
    sigma_drift: float = 1e-5
    sigma_noise: float = 5e-4
    aging_decay_rate: float = 1e-9


@dataclass
class MainPathConfig:
    distance_mm: float = 24.741
    half_angle_deg: float = 11.0
    water_transmission: float = 0.97
    interface_transmission: float = 0.90


@dataclass
class ReferencePathConfig:
    d1_mm: float = 1.294
    d2_mm: float = 3.126


@dataclass
class DetectorNoiseConfig:
    frac_noise: float = 0.002
    noise_floor: float = 200.0
    conv_time_ms: int = 800
    lsb_nW_cm2: float = 1.2
    dark_codes_rms: float = 1.0
    T_ref_C: float = 25.0
    temp_slope_per_C: float = 0.02
    temp_exp_per_C: float = 0.0


@dataclass
class GrowthConfig:
    initial_cells_ml: float = 1e3
    carrying_capacity_cells_ml: float = 1e9
    mu_mean: float = 0.0004
    lag_time_s: float = 7200.0
    alpha_att: float = 2.5 / 1e9
    inoculation_time_s: float = 2400.0
    t_min_C: float = 8.0
    t_max_C: float = 45.0
    t_opt_C: float = 37.0
    c_ratk: float = 0.2
    mu_sampling_sigma_frac: float = 1.0


@dataclass
class ThermalPlantConfig:
    start_temp_C: float = 30.0
    ambient_temp_C: float = 25.0
    # Fresh media sits at ambient, NOT at setpoint, so every dose cools the vessel.
    media_temp_C: float = 25.0
    # The heater couples into the liquid through the circulation loop. Model the
    # element as its OWN thermal node: with no flow the controller still powers
    # it (the sensor reads cold, so it demands heat) and the element STORES that
    # energy instead of delivering it. When flow resumes the stored heat dumps in
    # at once and the liquid OVERSHOOTS setpoint — the element is hot, not just
    # "on". Forcing power to zero would miss this entirely.
    heater_needs_circulation: bool = True
    element_heat_capacity_J_C: float = 8.0    # thermal mass of the element
    element_coupling_W_per_C: float = 8.0     # element -> liquid, only with flow
    element_loss_W_per_C: float = 0.05        # element -> ambient standing loss
    element_max_temp_C: float = 90.0          # thermal cutoff / self-limit
    target_temp_C: float = 37.0
    hysteresis_C: float = 0.3
    max_heater_power_W: float = 15.0
    power_ramp_steps: int = 30
    heat_loss_W_per_C: float = 0.5
    volume_mL: float = 50.0
    density_kg_m3: float = 1000.0
    heat_capacity_J_kg_C: float = 4186.0


@dataclass
class BubbleConfig:
    """Two physically distinct bubble populations plus an intermittent churn state.

    micro  — gas entrained by the STIRRER/aeration. Many per second, tiny optical
             effect, sub-sample lifetime. At a 1 Hz sample rate these alias into
             what looks like elevated white measurement noise: the right job for
             the Kalman's R, not for a despiker.
    large  — slugs carried past the window by CIRCULATION. Rare, big, and lasting
             several seconds, so they appear as coherent multi-sample excursions:
             genuine outliers a despiker must gate out. A per-sample sigma test
             won't see them as single-point spikes.
    churn  — a transient regime (foaming / gas slugging) that multiplies both
             rates and amplitudes for a while, then subsides. This is what breaks
             filters tuned on steady-state noise.
    """
    enabled: bool = True
    # Drive speeds (%), pushed live from the GUI motor fields.
    stirrer_speed: float = 100.0        # aeration  -> micro-bubble rate
    circulation_speed: float = 100.0    # circulation -> large-bubble rate
    # Below this circulation speed the loop does not actually move liquid, so no
    # fresh sample reaches the optical window: the reading goes STALE (it keeps
    # reporting whatever is sitting in the flow cell, not the vessel).
    circulation_threshold_pct: float = 50.0

    # --- micro: high frequency, low magnitude ---
    micro_lambda_max: float = 0.09      # events/s at 100% stirrer (reduced 10x)
    micro_frac_min: float = 2e-5
    micro_frac_max: float = 2.5e-4
    micro_width_s: float = 0.35         # << sample period -> aliases to noise

    # --- large: low frequency, high magnitude ---
    large_lambda_max: float = 0.0004    # events/s at 100% circulation (reduced 10x)
    large_frac_min: float = 3e-3
    large_frac_max: float = 2.5e-2
    large_width_s: float = 6.0          # spans ~6 samples at 1 Hz

    # --- churn transient ---
    transient_enabled: bool = True
    transient_lambda: float = 1.0 / 15000.0  # onset rate (events/s) (reduced 10x)
    transient_duration_s: float = 150.0
    transient_rate_gain: float = 5.0         # rate multiplier while churning
    transient_amp_gain: float = 2.5          # amplitude multiplier while churning

    affect_scatter: bool = True

    # --- channel coupling ---------------------------------------------------
    # Previously every bubble hit main and scatter with the IDENTICAL fraction
    # (correlation 1.0). That is not physical and it makes any main-vs-scatter
    # cross-check look far better in simulation than it can be on hardware:
    #   - a bubble settled ON the main PD is invisible to scatter
    #   - a bubble in the scatter path need not touch main
    #   - a shared-path bubble hits both, but with DIFFERENT magnitudes
    #     (different optical geometry / solid angle)
    # Each large bubble is drawn into one of these three cases.
    main_only_frac: float = 0.35        # P(bubble affects main only)
    scatter_only_frac: float = 0.25     # P(bubble affects scatter only)
    # remaining 0.40 -> shared, with scatter magnitude = main * U(gain_min, gain_max)
    shared_scatter_gain_min: float = 0.25
    shared_scatter_gain_max: float = 1.9


@dataclass
class ScatterConfig:
    baseline_main_signal_for_scatter: float = 6_000_000.0


@dataclass
class ModelConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    led: LEDConfig = field(default_factory=LEDConfig)
    main_path: MainPathConfig = field(default_factory=MainPathConfig)
    ref_path: ReferencePathConfig = field(default_factory=ReferencePathConfig)
    detector_noise: DetectorNoiseConfig = field(default_factory=DetectorNoiseConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    thermal: ThermalPlantConfig = field(default_factory=ThermalPlantConfig)
    bubbles: BubbleConfig = field(default_factory=BubbleConfig)
    scatter: ScatterConfig = field(default_factory=ScatterConfig)


# ============================== model layers ==============================

class LEDSourceModel:
    def __init__(self, config: LEDConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.log_drift = 0.0

    def nominal_power_mW(self, led_temp_C: float) -> float:
        cfg = self.config
        current_A = cfg.drive_current_mA / 1000.0
        power_W = cfg.slope_eff_W_per_A * current_A
        power_W *= 1.0 + cfg.temp_coeff_per_C * (led_temp_C - cfg.T_ref_C)
        return power_W * 1000.0

    def emitted_power_mW(self, led_temp_C: float, dt_s: float) -> float:
        cfg = self.config
        nominal = self.nominal_power_mW(led_temp_C)
        self.log_drift -= cfg.aging_decay_rate * dt_s
        self.log_drift += self.rng.normal(0.0, cfg.sigma_drift * np.sqrt(dt_s))
        fast_log_noise = self.rng.normal(0.0, cfg.sigma_noise)
        return nominal * np.exp(self.log_drift + fast_log_noise)


class OpticalPathModel:
    def __init__(self, main_path: MainPathConfig, ref_path: ReferencePathConfig):
        self.main_path = main_path
        self.ref_path = ref_path

    def main_irradiance_nW_cm2(self, led_power_mW: float) -> float:
        cfg = self.main_path
        d_m = cfg.distance_mm / 1000.0
        theta = np.deg2rad(cfg.half_angle_deg)
        beam_radius_m = d_m * np.tan(theta)
        area_m2 = np.pi * beam_radius_m**2
        transmitted_power_W = (led_power_mW / 1000.0) * cfg.water_transmission * cfg.interface_transmission
        irradiance_W_m2 = transmitted_power_W / area_m2
        return irradiance_W_m2 * 1e5

    def ref_irradiance_nW_cm2(self, led_power_mW: float) -> float:
        cfg = self.ref_path
        power_W = led_power_mW / 1000.0
        d1 = cfg.d1_mm / 1000.0
        d2 = cfg.d2_mm / 1000.0
        r_sq = d1**2 + d2**2
        irradiance_W_m2 = (power_W * d1) / (np.pi * r_sq ** (3.0 / 2.0))
        return irradiance_W_m2 * 1e5

    @staticmethod
    def transmission_from_cells(x_cells_ml: float, alpha_att: float) -> float:
        return float(np.exp(-alpha_att * x_cells_ml))


class ThermalPlant:
    def __init__(self, config: ThermalPlantConfig):
        self.config = config
        self.temperature_C = config.start_temp_C
        self.element_temp_C = config.start_temp_C   # heater element node
        self.heater_on = True
        self.heater_power_W = 0.0
        self.to_liquid_W = 0.0                      # power actually delivered
        self.ramp_steps_left = 0

    def mix_in(self, volume_mL: float, media_temp_C: float, vessel_mL: float) -> float:
        """Dose of fresh media at media_temp_C mixed into the working volume.
        Energy balance on perfect mixing: T <- (Vw*T + V*Tm) / (Vw + V).
        Media enters at ambient, so every dilution is also a cold shock."""
        if volume_mL <= 0 or vessel_mL <= 0:
            return self.temperature_C
        self.temperature_C = ((vessel_mL * self.temperature_C + volume_mL * media_temp_C)
                              / (vessel_mL + volume_mL))
        return self.temperature_C

    def step(self, dt_s: float, circulating: bool = True) -> float:
        cfg = self.config
        low = cfg.target_temp_C - cfg.hysteresis_C
        high = cfg.target_temp_C + cfg.hysteresis_C

        desired_heater_on = self.heater_on
        if self.temperature_C >= high:
            desired_heater_on = False
        elif self.temperature_C <= low:
            desired_heater_on = True

        if desired_heater_on != self.heater_on:
            self.heater_on = desired_heater_on
            self.ramp_steps_left = cfg.power_ramp_steps

        target_power = cfg.max_heater_power_W if self.heater_on else 0.0
        if self.ramp_steps_left > 0:
            alpha = 1.0 / self.ramp_steps_left
            self.heater_power_W = (1.0 - alpha) * self.heater_power_W + alpha * target_power
            self.ramp_steps_left -= 1
        else:
            self.heater_power_W = target_power

        # --- element node: absorbs the commanded power; only sheds it to the
        # liquid when there is flow to carry it. With the loop stopped the
        # element just gets HOT (bounded by its cutoff), banking energy.
        coupling = cfg.element_coupling_W_per_C
        if cfg.heater_needs_circulation and not circulating:
            coupling = 0.0
        self.to_liquid_W = coupling * (self.element_temp_C - self.temperature_C)
        elem_loss_W = cfg.element_loss_W_per_C * (self.element_temp_C - cfg.ambient_temp_C)
        self.element_temp_C += ((self.heater_power_W - self.to_liquid_W - elem_loss_W)
                                / max(cfg.element_heat_capacity_J_C, 1e-6)) * dt_s
        self.element_temp_C = min(self.element_temp_C, cfg.element_max_temp_C)

        # --- liquid node: gains only what the element can actually deliver.
        mass_kg = cfg.density_kg_m3 * cfg.volume_mL * 1e-6
        heat_loss_W = cfg.heat_loss_W_per_C * (self.temperature_C - cfg.ambient_temp_C)
        dTdt = (self.to_liquid_W - heat_loss_W) / (mass_kg * cfg.heat_capacity_J_kg_C)
        self.temperature_C += dTdt * dt_s
        return self.temperature_C


class BiomassProcessModel:
    def __init__(self, config: GrowthConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.cells_ml = config.initial_cells_ml
        self.inoculated = False
        self.inoculation_clock_s = 0.0

    def mu_temperature(self, temperature_C: float, mu_mean: Optional[float] = None) -> float:
        cfg = self.config
        if temperature_C <= cfg.t_min_C or temperature_C >= cfg.t_max_C:
            return 0.0
        reference_mu = cfg.mu_mean if mu_mean is None else mu_mean
        b_ratk = np.sqrt(reference_mu) / (cfg.t_opt_C - cfg.t_min_C)
        term = 1.0 - np.exp(cfg.c_ratk * (temperature_C - cfg.t_max_C))
        sqrt_mu = b_ratk * (temperature_C - cfg.t_min_C) * term
        return float(max(0.0, sqrt_mu * sqrt_mu))

    def sample_mu_mean(self) -> float:
        cfg = self.config
        sigma = abs(cfg.mu_mean) * cfg.mu_sampling_sigma_frac
        return float(max(self.rng.normal(cfg.mu_mean, sigma), 0.0))

    def step(self, absolute_time_s: float, temperature_C: float, dt_s: float) -> float:
        cfg = self.config
        if absolute_time_s < cfg.inoculation_time_s:
            return self.cells_ml
        if not self.inoculated:
            self.inoculated = True
            self.inoculation_clock_s = 0.0
        self.inoculation_clock_s += dt_s
        if self.inoculation_clock_s < cfg.lag_time_s:
            return self.cells_ml
        sampled_mu_mean = self.sample_mu_mean()
        mu_T = self.mu_temperature(temperature_C, mu_mean=sampled_mu_mean)
        dx = mu_T * self.cells_ml * (1.0 - self.cells_ml / cfg.carrying_capacity_cells_ml) * dt_s
        self.cells_ml += dx
        return self.cells_ml


class DetectorModel:
    def __init__(self, calibration: OpticalCalibration, optical_path: OpticalPathModel,
                 led: LEDSourceModel, config: ModelConfig, rng: np.random.Generator):
        self.calibration = calibration
        self.optical_path = optical_path
        self.led = led
        self.config = config
        self.rng = rng

        baseline_led_power = led.nominal_power_mW(config.thermal.ambient_temp_C)
        baseline_main = optical_path.main_irradiance_nW_cm2(baseline_led_power)
        baseline_ref = optical_path.ref_irradiance_nW_cm2(baseline_led_power)

        self.k_main = calibration.main_obs / baseline_main
        self.k_ref = calibration.ref_obs / baseline_ref

    @staticmethod
    def signal_from_irradiance(irradiance: float, gain: float) -> float:
        return gain * irradiance

    def add_opt3002_like_noise(self, signal: float, gain: float, temperature_C: float) -> float:
        cfg = self.config.detector_noise
        ct_scale = np.sqrt(800.0 / max(1.0, float(cfg.conv_time_ms)))
        lsb_units = gain * cfg.lsb_nW_cm2
        sigma_quant = 0.5 * lsb_units
        sigma_dark = cfg.dark_codes_rms * lsb_units
        sigma_base = ct_scale * np.sqrt(sigma_quant**2 + sigma_dark**2)

        dT = temperature_C - cfg.T_ref_C
        temp_multiplier = (1.0 + cfg.temp_slope_per_C * dT) * np.exp(cfg.temp_exp_per_C * dT)
        sigma_T = sigma_base * max(0.0, temp_multiplier)
        return signal + self.rng.normal(0.0, sigma_T)

    def add_generic_readout_noise(self, signal: float) -> float:
        cfg = self.config.detector_noise
        sigma = abs(signal) * cfg.frac_noise + cfg.noise_floor
        return signal + self.rng.normal(0.0, sigma)

    def measure_main(self, led_power_mW: float, transmission: float, detector_temp_C: float) -> float:
        irradiance = self.optical_path.main_irradiance_nW_cm2(led_power_mW)
        clean_signal = self.signal_from_irradiance(irradiance, self.k_main) * transmission
        signal = self.add_opt3002_like_noise(clean_signal, self.k_main, detector_temp_C)
        return self.add_generic_readout_noise(signal)

    def measure_ref(self, led_power_mW: float, detector_temp_C: float) -> float:
        irradiance = self.optical_path.ref_irradiance_nW_cm2(led_power_mW)
        clean_signal = self.signal_from_irradiance(irradiance, self.k_ref)
        signal = self.add_opt3002_like_noise(clean_signal, self.k_ref, detector_temp_C)
        return self.add_generic_readout_noise(signal)


class ScatterChannelModel:
    def __init__(self, calibration: OpticalCalibration, config: ModelConfig):
        self.baseline = calibration.scatter_obs
        self.gamma = 75_000.0 / (calibration.main_obs - 200_000.0)
        self.reference_main_signal = config.scatter.baseline_main_signal_for_scatter

    def signal(self, transmission: float) -> float:
        return self.baseline + self.gamma * self.reference_main_signal * (1.0 - transmission)


class BubbleDisturbanceModel:
    """Micro + large bubble populations with an intermittent churn transient.

    Each population is an independent Poisson process whose rate scales with its
    driving motor; events are Gaussian pulses in time. Keeping them separate (and
    reporting them separately) is what lets a filter be developed against the two
    failure modes independently rather than against one lumped 'noise' blob.
    """

    def __init__(self, config: BubbleConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.micro_events = []      # (t0, width_s, amp_fraction)
        self.large_events = []
        self.churn_until_s = -1.0   # transient active while current_time < this
        # Diagnostics for the last distortion_fraction() call.
        self.last_micro = 0.0
        self.last_large = 0.0
        self.last_main_frac = 0.0
        self.last_scatter_frac = 0.0

    @property
    def churning(self) -> bool:
        return self.churn_until_s > 0.0

    def _spawn(self, events, lam, dt_s, t, amp_lo, amp_hi, width, amp_gain):
        """Bernoulli-approximated Poisson arrival for one population.

        Each event gets INDEPENDENT main/scatter amplitudes: a bubble may sit on
        the main PD (scatter blind), sit in the scatter path (main blind), or be
        in the shared beam path and hit both by different amounts.
        """
        if lam <= 0.0:
            return
        p = 1.0 - np.exp(-lam * dt_s)
        if self.rng.random() < p:
            cfg = self.config
            amp = float(self.rng.uniform(amp_lo, amp_hi)) * amp_gain
            roll = self.rng.random()
            if roll < cfg.main_only_frac:
                a_main, a_scat = amp, 0.0
            elif roll < cfg.main_only_frac + cfg.scatter_only_frac:
                a_main, a_scat = 0.0, amp
            else:
                a_main = amp
                a_scat = amp * float(self.rng.uniform(cfg.shared_scatter_gain_min,
                                                      cfg.shared_scatter_gain_max))
            events.append((t, width, a_main, a_scat))

    def maybe_start_event(self, current_time_s: float, dt_s: float):
        cfg = self.config
        if not cfg.enabled:
            return

        # --- churn transient state machine ---
        if cfg.transient_enabled:
            if self.churn_until_s > 0.0 and current_time_s >= self.churn_until_s:
                self.churn_until_s = -1.0                      # subsided
            elif self.churn_until_s <= 0.0:
                p = 1.0 - np.exp(-cfg.transient_lambda * dt_s)
                if self.rng.random() < p:
                    self.churn_until_s = current_time_s + cfg.transient_duration_s
        rate_gain = cfg.transient_rate_gain if self.churning else 1.0
        amp_gain  = cfg.transient_amp_gain  if self.churning else 1.0

        # --- micro: stirrer-driven, high rate / low amplitude ---
        self._spawn(self.micro_events,
                    cfg.micro_lambda_max * (cfg.stirrer_speed / 100.0) * rate_gain,
                    dt_s, current_time_s,
                    cfg.micro_frac_min, cfg.micro_frac_max, cfg.micro_width_s, amp_gain)

        # --- large: circulation-driven, low rate / high amplitude ---
        self._spawn(self.large_events,
                    cfg.large_lambda_max * (cfg.circulation_speed / 100.0) * rate_gain,
                    dt_s, current_time_s,
                    cfg.large_frac_min, cfg.large_frac_max, cfg.large_width_s, amp_gain)

    @staticmethod
    def _sum_events(events, t):
        """Sum active Gaussian pulses, dropping expired ones.
        Returns (main_total, scatter_total, kept)."""
        m_tot = s_tot = 0.0
        kept = []
        for t0, width_s, a_main, a_scat in events:
            age = t - t0
            if age < 4.0 * width_s:
                shape = np.exp(-0.5 * (age / width_s) ** 2)
                m_tot += a_main * shape
                s_tot += a_scat * shape
                kept.append((t0, width_s, a_main, a_scat))
        return m_tot, s_tot, kept

    def distortion_fraction(self, current_time_s: float) -> float:
        """Refresh both channels; returns the MAIN fraction (back-compat).
        Use .last_main_frac / .last_scatter_frac for the per-channel values."""
        mm, ms, self.micro_events = self._sum_events(self.micro_events, current_time_s)
        lm, ls, self.large_events = self._sum_events(self.large_events, current_time_s)
        self.last_micro = mm + lm            # main-side micro+large (diagnostic)
        self.last_large = lm
        self.last_main_frac = mm + lm
        self.last_scatter_frac = ms + ls
        return self.last_main_frac

    def apply(self, clean_signal: float, current_time_s: float) -> float:
        """Back-compat: applies the MAIN-channel distortion."""
        return clean_signal * (1.0 + self.distortion_fraction(current_time_s))

    def apply_channels(self, main: float, scatter: float, t: float):
        """Apply the correct, INDEPENDENT distortion to each channel in one pass."""
        self.distortion_fraction(t)          # refreshes both fractions
        return (main * (1.0 + self.last_main_frac),
                scatter * (1.0 + self.last_scatter_frac))


class ReconstructionModel:
    """Twin's own OD reconstruction. The GUI ignores this and re-derives OD from
    the raw Main/Ref it emits, but it's kept so the twin is self-contained/testable."""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.beta = 0.0
        self._beta_samples = []

    def update_beta(self, time_s: float, main_signal: float, ref_signal: float):
        if time_s <= self.config.beta_warmup_s and ref_signal != 0:
            self._beta_samples.append(main_signal / ref_signal)
            self.beta = float(np.mean(self._beta_samples))

    def calculate_od(self, main_signal: float, ref_signal: float) -> float:
        eps = 1e-12
        beta = max(self.beta, eps)
        main = max(main_signal, eps)
        ref = max(ref_signal, eps)
        return float(-self.config.od_calibration_constant * np.log10(main / (ref * beta)))


# ============================== engine ==============================

class SensorTwinEngine:
    def __init__(self, calibration: OpticalCalibration, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        seed = self.config.simulation.random_seed
        self.rng = np.random.default_rng(seed)

        self.thermal = ThermalPlant(self.config.thermal)
        self.led = LEDSourceModel(self.config.led, self.rng)
        self.optical_path = OpticalPathModel(self.config.main_path, self.config.ref_path)
        self.growth = BiomassProcessModel(self.config.growth, self.rng)

        self.detector = DetectorModel(calibration, self.optical_path, self.led, self.config, self.rng)
        self.scatter = ScatterChannelModel(calibration, self.config)
        self.bubbles = BubbleDisturbanceModel(self.config.bubbles, self.rng)
        self.reconstruction = ReconstructionModel(self.config.simulation)

        self.time_s = 0.0
        # Cells actually sitting in the optical window. Tracks the vessel while
        # liquid is circulating; FREEZES when flow stops, because nothing new
        # reaches the sensor.
        self.window_cells_ml = None
        self.circulating = True

    def add_fresh_media(self, volume_mL: float, vessel_mL: float,
                        media_temp_C: Optional[float] = None) -> None:
        """One dilution dose. Does BOTH things a real dose does: displaces cells
        AND drops the temperature, because the media is at ambient, not setpoint."""
        if volume_mL <= 0 or vessel_mL <= 0:
            return
        self.growth.cells_ml *= vessel_mL / (vessel_mL + volume_mL)
        tm = self.config.thermal.media_temp_C if media_temp_C is None else media_temp_C
        self.thermal.mix_in(volume_mL, tm, vessel_mL)

    # ---- real-time coupling hooks (called by Comms.simulate_twin) ----
    def set_target_temp(self, target_C: float) -> None:
        try:
            self.config.thermal.target_temp_C = float(target_C)
        except (TypeError, ValueError):
            pass

    def set_motor_speeds(self, stirrer: Optional[float] = None,
                         circulation: Optional[float] = None) -> None:
        """Live-couple the drive speeds: stirrer/aeration sets the micro-bubble
        rate, circulation sets the large-bubble (slug) rate."""
        b = self.config.bubbles
        try:
            if stirrer is not None:
                b.stirrer_speed = float(stirrer)
            if circulation is not None:
                b.circulation_speed = float(circulation)
        except (TypeError, ValueError):
            pass

    def set_motor_speed(self, motor_speed: float) -> None:
        """Back-compat: treat a lone speed as the stirrer (micro-bubble driver)."""
        self.set_motor_speeds(stirrer=motor_speed)

    def step(self) -> Dict[str, float]:
        dt = self.config.simulation.dt_s
        self.time_s += dt

        # Is the loop actually moving liquid? Below the threshold the pump does
        # not shift sample, so the sensor sees stale liquid and the heater cannot
        # couple into the bulk.
        b = self.config.bubbles
        self.circulating = b.circulation_speed >= b.circulation_threshold_pct

        liquid_temp_C = self.thermal.step(dt, circulating=self.circulating)
        led_power_mW = self.led.emitted_power_mW(liquid_temp_C, dt)
        cells_ml = self.growth.step(self.time_s, liquid_temp_C, dt)

        # The optics only ever see what is IN the window. With flow that tracks the
        # vessel. Without flow the trapped slug is its own tiny batch culture: the
        # cells in it KEEP GROWING — it is not sterile — but at the LOCAL
        # temperature, which is decaying toward ambient because the heater can no
        # longer couple. So OD creeps upward, ever more slowly, rather than truly
        # freezing; and it no longer reflects the vessel or responds to dilution.
        if self.circulating or self.window_cells_ml is None:
            self.window_cells_ml = cells_ml
        else:
            g = self.config.growth
            if self.growth.inoculated and self.growth.inoculation_clock_s >= g.lag_time_s:
                mu_T = self.growth.mu_temperature(liquid_temp_C)
                self.window_cells_ml += (mu_T * self.window_cells_ml
                                         * (1.0 - self.window_cells_ml / g.carrying_capacity_cells_ml)
                                         * dt)
        transmission = self.optical_path.transmission_from_cells(
            self.window_cells_ml, self.config.growth.alpha_att)

        self.bubbles.maybe_start_event(self.time_s, dt)
        main_signal = self.detector.measure_main(led_power_mW, transmission, liquid_temp_C)
        ref_signal = self.detector.measure_ref(led_power_mW, self.config.thermal.ambient_temp_C)
        scatter_signal = self.scatter.signal(transmission)

        # Independent per-channel distortion (a bubble on the main PD is invisible
        # to scatter, and vice versa) — one pass so both share the same event set.
        if self.config.bubbles.affect_scatter:
            main_signal, scatter_signal = self.bubbles.apply_channels(
                main_signal, scatter_signal, self.time_s)
        else:
            main_signal = self.bubbles.apply(main_signal, self.time_s)

        self.reconstruction.update_beta(self.time_s, main_signal, ref_signal)
        od = self.reconstruction.calculate_od(main_signal, ref_signal)

        # Ground-truth OD: what a perfect, noise-free, bubble-free instrument would
        # read for this cell density. The whole point of a twin — filters can be
        # scored against this instead of against each other.
        od_true = float(-self.config.simulation.od_calibration_constant
                        * np.log10(max(transmission, 1e-12)))

        return {
            "time_s": self.time_s,
            "temperature_C": liquid_temp_C,
            "led_power_mW": led_power_mW,
            "cells_ml": cells_ml,                 # bulk vessel
            "window_cells_ml": self.window_cells_ml,   # what the sensor sees
            "circulating": 1.0 if self.circulating else 0.0,
            "element_temp_C": self.thermal.element_temp_C,
            "heater_to_liquid_W": self.thermal.to_liquid_W,
            "transmission": transmission,
            "main_signal": main_signal,
            "ref_signal": ref_signal,
            "scatter_signal": scatter_signal,
            "beta": self.reconstruction.beta,
            "od": od,
            "od_true": od_true,
            # Disturbance diagnostics — lets a filter be evaluated per failure mode.
            "bubble_micro": self.bubbles.last_micro,
            "bubble_large": self.bubbles.last_large,
            "bubble_main": self.bubbles.last_main_frac,
            "bubble_scatter": self.bubbles.last_scatter_frac,
            "churning": 1.0 if self.bubbles.churning else 0.0,
        }

    def run(self, steps: Optional[int] = None):
        """Batch run — returns a list of row dicts (no pandas dependency)."""
        n_steps = self.config.simulation.max_steps if steps is None else steps
        return [self.step() for _ in range(n_steps)]


def od_ceiling(od_calibration_constant: float, alpha_att: float,
               carrying_capacity_cells_ml: float) -> float:
    """OD the twin saturates at, given the optics + biology:

        transmission_min = exp(-alpha_att * carrying_capacity)
        OD_max = -C * log10(transmission_min) = C * alpha_att * cap / ln(10)

    Growth rate (mu_mean) only changes how fast you get there, never how high.
    """
    return float(od_calibration_constant * alpha_att * carrying_capacity_cells_ml / np.log(10.0))


def build_engine(target_temp_C: float = 37.0, seed: int = 1,
                 mu_mean: Optional[float] = None,
                 inoculation_time_s: Optional[float] = None,
                 lag_time_s: Optional[float] = None,
                 carrying_capacity_cells_ml: Optional[float] = None,
                 alpha_att: Optional[float] = None,
                 od_calibration_constant: Optional[float] = None,
                 bubbles_enabled: Optional[bool] = None,
                 bubble_micro_rate: Optional[float] = None,
                 bubble_large_rate: Optional[float] = None,
                 bubble_churn_rate: Optional[float] = None) -> SensorTwinEngine:
    """Construct an engine with the usual defaults, optionally overriding the
    optics/growth knobs most useful for a live demo.

    od_calibration_constant should be passed the SAME value the consumer uses to
    reconstruct OD (the GUI's [Offsets] CalibrationConst), otherwise the twin's
    own reported `od` silently diverges from the OD the GUI computes from the
    twin's raw Main/Ref."""
    calibration = OpticalCalibration()
    config = ModelConfig()
    config.thermal.target_temp_C = float(target_temp_C)
    config.simulation.random_seed = int(seed)
    if od_calibration_constant is not None:
        config.simulation.od_calibration_constant = float(od_calibration_constant)
    g = config.growth
    if mu_mean is not None:                     g.mu_mean = float(mu_mean)
    if inoculation_time_s is not None:          g.inoculation_time_s = float(inoculation_time_s)
    if lag_time_s is not None:                  g.lag_time_s = float(lag_time_s)
    if carrying_capacity_cells_ml is not None:  g.carrying_capacity_cells_ml = float(carrying_capacity_cells_ml)
    if alpha_att is not None:                   g.alpha_att = float(alpha_att)
    if bubbles_enabled is not None:             config.bubbles.enabled = bool(bubbles_enabled)
    b = config.bubbles
    if bubble_micro_rate is not None:           b.micro_lambda_max = float(bubble_micro_rate)
    if bubble_large_rate is not None:           b.large_lambda_max = float(bubble_large_rate)
    if bubble_churn_rate is not None:           b.transient_lambda = float(bubble_churn_rate)
    return SensorTwinEngine(calibration=calibration, config=config)


if __name__ == "__main__":
    # Quick standalone smoke test (no GUI).
    eng = build_engine(target_temp_C=37.0, seed=1, mu_mean=0.0008,
                       inoculation_time_s=300, lag_time_s=600)
    rows = eng.run(steps=2000)
    last = rows[-1]
    print(f"t={last['time_s']:.0f}s  T={last['temperature_C']:.2f}C  "
          f"cells={last['cells_ml']:.0f}/mL  main={last['main_signal']:.3e}  "
          f"ref={last['ref_signal']:.3e}  od={last['od']:.4f}")
