"""
Week 1 Day 4 — Generate 4 additional fault scenario datasets and evaluate
the LSTM autoencoder against all 5 faults (battery + 4 new).

Historical references:
  - O2 leak       : ISS microleak incidents (Zvezda module, 2019–2021)
  - CO2 scrubber  : Apollo 13 CO2 buildup crisis (April 1970)
  - Thermal spike : Multiple Space Shuttle thermal control system anomalies
  - Power spike   : Generic sudden electrical fault (no-drift baseline)

Run from project root:
    python data/generate_fault_scenarios.py
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib

from data.generator import SpacecraftTelemetryGenerator

SENSORS    = ['cabin_temperature', 'oxygen_percentage', 'battery_percentage', 'co2_percentage']
WINDOW_SIZE = 60
THRESHOLD   = 0.053946    # calibrated in Day 2 (normal mean + 3 sigma)

# ── All 5 fault scenarios (including Day 2 battery baseline) ──────────────────
FAULT_SCENARIOS = {
    'battery': {
        'label':          'Battery Cell Collapse',
        'history':        'Day 2 baseline fault',
        'csv':            'fault_telemetry.csv',      # already generated in Day 1
        'sensor':         'battery_percentage',
        'fault_type':     'slow_drift',
        'fault_start':    4000,
        'drift_duration': 120,
        'magnitude':      -80.0,
        'danger_thresh':  '< 15 %',
    },
    'oxygen': {
        'label':          'O2 Pressure Leak',
        'history':        'ISS Zvezda module microleak (2019-21)',
        'csv':            'oxygen_fault.csv',
        'sensor':         'oxygen_percentage',
        'fault_type':     'slow_drift',
        'fault_start':    5000,
        'drift_duration': 90,           # 90-min leak before hypoxia threshold
        'magnitude':      -1.5,         # 20.75% -> ~19.25%, crosses 19.5% danger at ~80 min
        'danger_thresh':  '< 19.5 %',
    },
    'co2': {
        'label':          'CO2 Scrubber Failure',
        'history':        'Apollo 13 CO2 crisis (April 1970)',
        'csv':            'co2_fault.csv',
        'sensor':         'co2_percentage',
        'fault_type':     'slow_drift',
        'fault_start':    6000,
        'drift_duration': 120,          # 2-hr rise mirrors Apollo 13 timeline
        'magnitude':      1.1,          # ~0.45% -> ~1.55%, crosses 1.0% danger at ~70 min
        'danger_thresh':  '> 1.0 %',
    },
    'temperature': {
        'label':          'Thermal Control Failure',
        'history':        'Space Shuttle thermal control anomalies',
        'csv':            'temperature_fault.csv',
        'sensor':         'cabin_temperature',
        'fault_type':     'slow_drift',
        'fault_start':    7000,
        'drift_duration': 90,           # 90-min ramp, mirrors real TCS failure timescales
        'magnitude':      25.0,         # 72°F -> ~97°F, crosses 85°F danger at ~47 min
        'danger_thresh':  '> 85 °F',
    },
    'power': {
        'label':          'Power System Spike',
        'history':        'Sudden electrical failure (no drift)',
        'csv':            'power_spike_fault.csv',
        'sensor':         'cabin_temperature',
        'fault_type':     'sudden_spike',
        'fault_start':    8000,
        'drift_duration': 0,            # instant — no warning window
        'magnitude':      40.0,         # 72°F -> ~112°F, immediate thermal spike
        'danger_thresh':  'Instant failure',
    },
}


# ── Step 1: Generate the 4 new fault datasets ─────────────────────────────────
def generate_datasets():
    print("=" * 65)
    print("Step 1  —  Generate fault datasets")
    print("=" * 65)

    gen     = SpacecraftTelemetryGenerator(seed=42)
    base_df = gen.generate()
    print(f"  Base telemetry: {len(base_df):,} rows (10 days @ 1 min)")

    data_dir = ROOT / 'data'
    generated = 0

    for key, sc in FAULT_SCENARIOS.items():
        if key == 'battery':
            print(f"  [SKIP] battery — already exists as fault_telemetry.csv (Day 1)")
            continue

        df = gen.inject_fault(
            base_df,
            sensor_name=sc['sensor'],
            fault_start_time=sc['fault_start'],
            fault_type=sc['fault_type'],
            magnitude=sc['magnitude'],
            drift_duration=sc['drift_duration'],
        )
        out = data_dir / sc['csv']
        df.to_csv(out, index=False)

        # Sanity: show sensor value at fault_start, midpoint, and end
        mid = sc['fault_start'] + sc['drift_duration'] // 2
        end = sc['fault_start'] + sc['drift_duration']
        v0  = base_df.loc[base_df['minute'] == sc['fault_start'], sc['sensor']].values[0]
        vm  = df.loc[df['minute'] == min(mid, 14399), sc['sensor']].values[0]
        ve  = df.loc[df['minute'] == min(end, 14399), sc['sensor']].values[0]
        print(f"  [{key:12s}] {sc['sensor']:25s}  "
              f"start={v0:.3f}  mid={vm:.3f}  end={ve:.3f}  -> {out.name}")
        generated += 1

    print(f"  Generated {generated} new datasets.")


# ── Step 2: LSTM evaluation helpers ──────────────────────────────────────────
def create_windows(data: np.ndarray, ws: int) -> np.ndarray:
    n   = len(data) - ws + 1
    idx = np.arange(ws)[None, :] + np.arange(n)[:, None]
    return data[idx]


def evaluate_fault(scenario_key: str, model, scaler) -> dict:
    sc       = FAULT_SCENARIOS[scenario_key]
    csv_path = ROOT / 'data' / sc['csv']
    df       = pd.read_csv(csv_path, parse_dates=['timestamp'])

    # Load normal data for threshold calibration (already done in Day 2, reuse threshold)
    X_scaled  = scaler.transform(df[SENSORS].values.astype(np.float32))
    W         = create_windows(X_scaled, WINDOW_SIZE)
    recon     = model.predict(W, batch_size=256, verbose=0)
    errors_f  = np.mean((W - recon) ** 2, axis=(1, 2))

    # Pad to align with minute indices
    pad    = np.full(WINDOW_SIZE - 1, np.nan)
    errors = np.concatenate([pad, errors_f])
    flags  = np.concatenate([np.zeros(WINDOW_SIZE - 1, dtype=int),
                              (errors_f > THRESHOLD).astype(int)])

    fault_start = sc['fault_start']
    fault_end   = fault_start + sc['drift_duration']

    # First detection AT OR AFTER fault_start
    in_fault = (df['minute'].values >= fault_start) & (flags == 1)
    if in_fault.any():
        fd  = int(df['minute'].values[in_fault][0])
        mbf = fault_end - fd   # minutes before "full failure"; negative = post-failure
    else:
        fd  = None
        mbf = None

    return {
        'scenario':              scenario_key,
        'label':                 sc['label'],
        'history':               sc['history'],
        'sensor':                sc['sensor'],
        'fault_type':            sc['fault_type'],
        'fault_start':           fault_start,
        'fault_end':             fault_end,
        'drift_duration':        sc['drift_duration'],
        'danger_thresh':         sc['danger_thresh'],
        'first_detection':       fd,
        'minutes_before_failure': mbf,
        'errors':                errors,
        'flags':                 flags,
    }


# ── Step 3: Run LSTM on all 5 faults ─────────────────────────────────────────
def run_lstm_evaluation():
    print("\n" + "=" * 65)
    print("Step 2  —  Load LSTM model and evaluate all 5 faults")
    print("=" * 65)

    import keras
    print(f"  Keras {keras.__version__}  |  backend: {keras.backend.backend()}")

    model  = keras.models.load_model(str(ROOT / 'models' / 'lstm_autoencoder.keras'))
    scaler = joblib.load(ROOT / 'models' / 'lstm_scaler.pkl')
    print(f"  Loaded: lstm_autoencoder.keras  +  lstm_scaler.pkl")
    print(f"  Detection threshold: {THRESHOLD:.5f}")

    results = []
    for key in FAULT_SCENARIOS:
        print(f"  Evaluating: {key} ...")
        r = evaluate_fault(key, model, scaler)
        results.append(r)
        fd_str  = f"min {r['first_detection']}" if r['first_detection'] else "NOT DETECTED"
        mbf_str = (f"{r['minutes_before_failure']} min"
                   if r['minutes_before_failure'] is not None else "N/A")
        print(f"    -> {fd_str}   ({mbf_str} before full failure)")

    return results


# ── Step 4: Print results table ───────────────────────────────────────────────
def print_results_table(results: list):
    DIV = "=" * 95

    lines = []
    lines.append(DIV)
    lines.append("  ORION SPACECRAFT  —  DAY 4  —  ALL FAULT SCENARIOS  —  LSTM DETECTION RESULTS")
    lines.append(DIV)
    lines.append(
        f"  {'Scenario':<28}  {'Sensor':<22}  {'Fault Start':>11}  "
        f"{'Drift':>6}  {'1st Detection':>14}  {'Warning':>10}  {'Danger'}",
    )
    lines.append("  " + "-" * 91)

    for r in results:
        fd  = f"min {r['first_detection']}" if r['first_detection'] else "NOT DETECTED"
        if r['minutes_before_failure'] is not None:
            mbf = r['minutes_before_failure']
            if mbf >= 0:
                mbf_str = f"{mbf} min"
            else:
                mbf_str = f"+{abs(mbf)} min post"
        else:
            mbf_str = "N/A"

        drift_str = f"{r['drift_duration']} min" if r['drift_duration'] > 0 else "instant"

        lines.append(
            f"  {r['label']:<28}  {r['sensor']:<22}  {r['fault_start']:>11}  "
            f"  {drift_str:>8}  {fd:>14}  {mbf_str:>10}  {r['danger_thresh']}"
        )

    lines.append(DIV)
    lines.append("")
    lines.append("  SUMMARY")
    lines.append("  " + "-" * 50)

    for r in results:
        if r['first_detection'] and r['minutes_before_failure'] is not None:
            mbf = r['minutes_before_failure']
            if mbf > 0:
                lines.append(f"  * {r['label']:<28}  detected {mbf} min before failure  "
                             f"(min {r['first_detection']})")
            else:
                lines.append(f"  * {r['label']:<28}  detected {abs(mbf)} min AFTER fault  "
                             f"(min {r['first_detection']})  [instant fault — no warning window]")
        else:
            lines.append(f"  * {r['label']:<28}  NOT DETECTED")

    lines.append("")
    lines.append("  Historical references:")
    for r in results:
        lines.append(f"    {r['label']:<28}  — {r['history']}")
    lines.append(DIV)

    output = "\n".join(lines)
    print("\n" + output)

    out_path = ROOT / 'docs' / 'day4_all_faults_results.txt'
    out_path.write_text(output, encoding='utf-8')
    print(f"\nSaved: docs/day4_all_faults_results.txt")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    generate_datasets()
    results = run_lstm_evaluation()
    print_results_table(results)
    print("\nDay 4 data generation and evaluation complete.")


if __name__ == '__main__':
    main()
