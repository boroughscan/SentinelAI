import numpy as np
import pandas as pd


class SpacecraftTelemetryGenerator:
    """Simulates Orion spacecraft sensor telemetry with realistic physics-based patterns."""

    # How far each sensor deviates when it fully fails.
    # Signs matter: temperature/CO2 spike UP, oxygen/battery collapse DOWN.
    FAULT_MAGNITUDES = {
        'cabin_temperature': 30.0,    # +30°F: thermal control failure → overheating
        'oxygen_percentage': -5.0,    # -5%:   O2 recycler failure or slow leak
        'battery_percentage': -80.0,  # -80%:  battery cell collapse → drains to 0
        'co2_percentage':     1.5,    # +1.5%: CO2 scrubber failure → toxic buildup
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.duration_minutes = 10 * 24 * 60  # 14,400 one-minute readings over 10 days

    # ------------------------------------------------------------------ #
    #  Private sensor generators — all fully vectorized (no Python loops) #
    # ------------------------------------------------------------------ #

    def _generate_cabin_temperature(self, n: int) -> np.ndarray:
        minutes = np.arange(n)
        # Spacecraft interior warms during crew activity hours, cools in sleep windows.
        # One full oscillation every 1440 minutes (24 h), amplitude ±2°F around 72°F.
        day_night = 2.0 * np.sin(2 * np.pi * minutes / 1440)
        noise = self.rng.normal(0, 0.3, n)
        return 72.0 + day_night + noise

    def _generate_oxygen_percentage(self, n: int) -> np.ndarray:
        # Slow 3-day oscillation models crew consumption vs. electrolysis recycling cadence.
        slow_var = 0.1 * np.sin(2 * np.pi * np.arange(n) / (1440 * 3))
        noise = self.rng.normal(0, 0.02, n)
        return 20.75 + slow_var + noise

    def _generate_battery_percentage(self, n: int) -> np.ndarray:
        # 8-hour discharge (eclipse + high ops load) then 2-hour recharge (solar exposure).
        # This gives ~24 realistic charge cycles over the 10-day window.
        discharge_time = 480   # minutes
        charge_time    = 120   # minutes
        cycle_time     = discharge_time + charge_time   # 600 min = 10 h

        pos = np.arange(n) % cycle_time
        values = np.where(
            pos < discharge_time,
            100.0 - (80.0 * pos / discharge_time),          # 100% → 20% over 8 h
            20.0  + (80.0 * (pos - discharge_time) / charge_time),  # 20% → 100% over 2 h
        )
        return np.clip(values + self.rng.normal(0, 0.5, n), 0.0, 100.0)

    def _generate_co2_percentage(self, n: int) -> np.ndarray:
        # Crew breathing raises CO2 over 6 h; scrubber activates and clears it in 30 min.
        rise_time  = 360   # minutes
        scrub_time = 30    # minutes
        cycle_time = rise_time + scrub_time  # 390 min ≈ 6.5 h

        pos = np.arange(n) % cycle_time
        values = np.where(
            pos < rise_time,
            0.3 + (0.3 * pos / rise_time),                    # 0.3% → 0.6% over 6 h
            0.6 - (0.3 * (pos - rise_time) / scrub_time),     # 0.6% → 0.3% over 30 min
        )
        return np.clip(values + self.rng.normal(0, 0.005, n), 0.0, 1.0)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(self) -> pd.DataFrame:
        """
        Generate 10 days of 1-minute telemetry for all four sensors.

        Returns a DataFrame with columns:
            timestamp, minute, cabin_temperature, oxygen_percentage,
            battery_percentage, co2_percentage
        """
        n = self.duration_minutes
        return pd.DataFrame({
            'timestamp':         pd.date_range(start='2025-01-01', periods=n, freq='1min'),
            'minute':            np.arange(n),
            'cabin_temperature': self._generate_cabin_temperature(n),
            'oxygen_percentage': self._generate_oxygen_percentage(n),
            'battery_percentage':self._generate_battery_percentage(n),
            'co2_percentage':    self._generate_co2_percentage(n),
        })

    def inject_fault(
        self,
        df: pd.DataFrame,
        sensor_name: str,
        fault_start_time: int,
        fault_type: str,
        magnitude: float = None,
        drift_duration: int = 120,
    ) -> pd.DataFrame:
        """
        Inject a hardware fault into one sensor column.

        Args:
            df               : DataFrame produced by generate()
            sensor_name      : which sensor to corrupt
            fault_start_time : minute index where the fault begins
            fault_type       : 'slow_drift'   — linear ramp to full failure over drift_duration.
                               'sudden_spike' — full-magnitude failure at fault_start_time.
            magnitude        : total sensor deviation at full failure.  Defaults to
                               FAULT_MAGNITUDES[sensor_name] if not supplied.
            drift_duration   : minutes for the slow_drift ramp (default 120 = 2 hours).

        Returns a copy of df with the fault applied — original is never modified.
        """
        if sensor_name not in self.FAULT_MAGNITUDES:
            raise ValueError(
                f"Unknown sensor '{sensor_name}'. "
                f"Valid sensors: {list(self.FAULT_MAGNITUDES)}"
            )
        if fault_type not in ('slow_drift', 'sudden_spike'):
            raise ValueError(
                f"fault_type must be 'slow_drift' or 'sudden_spike', got '{fault_type}'"
            )

        df        = df.copy()
        n         = len(df)
        if magnitude is None:
            magnitude = self.FAULT_MAGNITUDES[sensor_name]
        delta     = np.zeros(n)

        if fault_type == 'slow_drift':
            ramp_end        = min(fault_start_time + drift_duration, n)
            ramp_idx        = np.arange(fault_start_time, ramp_end)
            delta[ramp_idx] = magnitude * (ramp_idx - fault_start_time) / drift_duration
            delta[ramp_end:] = magnitude

        else:  # sudden_spike
            delta[fault_start_time:] = magnitude

        df[sensor_name] = df[sensor_name].values + delta
        return df
