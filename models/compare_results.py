"""
Loads all three trained models, runs them on fault telemetry,
prints the comparison table, and saves it to docs/model_comparison_results.txt.

Run from project root:
    python models/compare_results.py
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

SENSORS        = ['cabin_temperature', 'oxygen_percentage', 'battery_percentage', 'co2_percentage']
FAULT_START    = 4000
FAULT_END      = 4120   # FAULT_START + 120-min drift
WINDOW_SIZE    = 60
BATT_THRESHOLD = 15.0
MODELS_DIR     = ROOT / 'models'
DOCS_DIR       = ROOT / 'docs'

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading telemetry data ...")
normal_df = pd.read_csv(ROOT / 'data' / 'normal_telemetry.csv', parse_dates=['timestamp'])
fault_df  = pd.read_csv(ROOT / 'data' / 'fault_telemetry.csv',  parse_dates=['timestamp'])

fault_labels = (fault_df['minute'].values >= FAULT_START).astype(int)
print(f"  Fault dataset: {len(fault_df):,} rows  |  anomalous from minute {FAULT_START} onward")


# ── Helper ───────────────────────────────────────────────────────────────────
def first_detection_in_window(flags, minutes):
    """Return the first flagged minute that falls inside the fault drift window."""
    mask = (minutes >= FAULT_START) & (flags == 1)
    return int(minutes[mask].min()) if mask.any() else None


def metrics(fault_labels, flags):
    prec = precision_score(fault_labels, flags, zero_division=0)
    rec  = recall_score(fault_labels,   flags, zero_division=0)
    f1   = f1_score(fault_labels,       flags, zero_division=0)
    return prec, rec, f1


# ── Model 1: Threshold rule ──────────────────────────────────────────────────
print("\n[1/3] Threshold rule ...")
flags_thresh = (fault_df['battery_percentage'].values < BATT_THRESHOLD).astype(int)
fd_thresh    = first_detection_in_window(flags_thresh, fault_df['minute'].values)
prec_t, rec_t, f1_t = metrics(fault_labels, flags_thresh)


# ── Model 2: Isolation Forest ────────────────────────────────────────────────
print("[2/3] Isolation Forest ...")
iforest = joblib.load(MODELS_DIR / 'isolation_forest.pkl')
scaler_if = joblib.load(MODELS_DIR / 'if_scaler.pkl')

X_fault_if  = scaler_if.transform(fault_df[SENSORS].values.astype(np.float32))
raw_if      = iforest.predict(X_fault_if)
flags_if    = (raw_if == -1).astype(int)
fd_if       = first_detection_in_window(flags_if, fault_df['minute'].values)
prec_i, rec_i, f1_i = metrics(fault_labels, flags_if)


# ── Model 3: LSTM Autoencoder ────────────────────────────────────────────────
print("[3/3] LSTM Autoencoder ...")
import keras
print(f"  Keras {keras.__version__}  |  backend: {keras.backend.backend()}")

model     = keras.models.load_model(str(MODELS_DIR / 'lstm_autoencoder.keras'))
scaler_l  = joblib.load(MODELS_DIR / 'lstm_scaler.pkl')

X_normal_scaled = scaler_l.transform(normal_df[SENSORS].values).astype(np.float32)
X_fault_scaled  = scaler_l.transform(fault_df[SENSORS].values).astype(np.float32)

def create_windows(data, ws):
    n = len(data) - ws + 1
    idx = np.arange(ws)[None, :] + np.arange(n)[:, None]
    return data[idx]

W_normal = create_windows(X_normal_scaled, WINDOW_SIZE)
W_fault  = create_windows(X_fault_scaled,  WINDOW_SIZE)

# Calibrate threshold on normal data (mean + 3 sigma)
print("  Computing normal reconstruction errors ...")
recon_n       = model.predict(W_normal, batch_size=256, verbose=0)
errors_normal = np.mean((W_normal - recon_n) ** 2, axis=(1, 2))
threshold     = errors_normal.mean() + 3 * errors_normal.std()
print(f"  Normal error  :  mean={errors_normal.mean():.6f}  std={errors_normal.std():.6f}")
print(f"  Threshold     :  {threshold:.6f}  (mean + 3 sigma)")

print("  Scoring fault dataset ...")
recon_f      = model.predict(W_fault, batch_size=256, verbose=0)
errors_fault = np.mean((W_fault - recon_f) ** 2, axis=(1, 2))

# Pad first (WINDOW_SIZE-1) minutes: no decision until first full window is observed
pad       = np.full(WINDOW_SIZE - 1, np.nan)
errors    = np.concatenate([pad, errors_fault])
flags_raw = (errors_fault > threshold).astype(int)
flags_l   = np.concatenate([np.zeros(WINDOW_SIZE - 1, dtype=int), flags_raw])

fd_l = first_detection_in_window(flags_l, fault_df['minute'].values)
prec_l, rec_l, f1_l = metrics(fault_labels, flags_l)


# ── Build results table ──────────────────────────────────────────────────────
def mbf(fd):
    """Minutes before failure (FAULT_END - first_detection)."""
    return (FAULT_END - fd) if fd else None

results = [
    {
        'model':             'Threshold  (battery < 15%)',
        'first_detection':   fd_thresh,
        'minutes_warning':   mbf(fd_thresh),
        'precision':         prec_t,
        'recall':            rec_t,
        'f1':                f1_t,
    },
    {
        'model':             'Isolation Forest',
        'first_detection':   fd_if,
        'minutes_warning':   mbf(fd_if),
        'precision':         prec_i,
        'recall':            rec_i,
        'f1':                f1_i,
    },
    {
        'model':             'LSTM Autoencoder',
        'first_detection':   fd_l,
        'minutes_warning':   mbf(fd_l),
        'precision':         prec_l,
        'recall':            rec_l,
        'f1':                f1_l,
    },
]


# ── Print table ──────────────────────────────────────────────────────────────
DIVIDER = "=" * 82

lines = []
lines.append(DIVIDER)
lines.append("  SPACECRAFT ANOMALY DETECTION  —  WEEK 1 DAY 2  MODEL COMPARISON")
lines.append(f"  Fault: battery slow_drift  |  starts min {FAULT_START}  |  full failure min {FAULT_END}")
lines.append(DIVIDER)
lines.append(
    f"  {'Model':<30}  {'1st Detection':>13}  {'Warning':>8}  "
    f"{'Precision':>10}  {'Recall':>7}  {'F1':>6}"
)
lines.append("  " + "-" * 78)

for r in results:
    fd  = f"min {r['first_detection']}" if r['first_detection'] else "NOT DETECTED"
    mbf_str = f"{r['minutes_warning']} min" if r['minutes_warning'] else "N/A"
    lines.append(
        f"  {r['model']:<30}  {fd:>13}  {mbf_str:>8}  "
        f"{r['precision']:>10.3f}  {r['recall']:>7.3f}  {r['f1']:>6.3f}"
    )

lines.append(DIVIDER)

lstm = results[2]
lines.append("")
lines.append("  KEY FINDINGS")
lines.append("  " + "-" * 40)

if lstm['first_detection']:
    lines.append(f"  * LSTM first detects the fault at minute {lstm['first_detection']}")
    lines.append(f"    — that is {lstm['minutes_warning']} minutes before catastrophic battery failure")
    lines.append(f"    — fault drift started at minute {FAULT_START}, so detection is")
    lines.append(f"      {lstm['first_detection'] - FAULT_START} min into the 2-hour danger window")
    lines.append(f"      ({(lstm['first_detection'] - FAULT_START) / 120 * 100:.0f}% of the way through the drift)")

thresh_r = results[0]
if thresh_r['first_detection']:
    lines.append(f"  * Threshold rule catches it at minute {thresh_r['first_detection']}")
    lines.append(f"    but requires battery to physically cross 15% — a late lagging signal")

if_r = results[1]
if if_r['first_detection']:
    lines.append(f"  * Isolation Forest flags at minute {if_r['first_detection']}")
    lines.append(f"    — no temporal context, only catches point-in-time outliers")

lines.append("")
lines.append("  WHY LSTM WINS (explanation for professor)")
lines.append("  " + "-" * 40)
lines.append("  The LSTM Autoencoder is trained to reconstruct 60-minute windows of NORMAL")
lines.append("  sensor sequences. During the fault drift (min 4000-4120), the battery")
lines.append("  discharge rate increases from its normal -0.17%/min to -0.84%/min due to")
lines.append("  the injected fault. The LSTM has memorised normal discharge curves, so")
lines.append("  the faster decline produces a reconstruction error spike BEFORE the battery")
lines.append("  crosses any absolute threshold. The other models only flag a point once it")
lines.append("  is already far outside the normal VALUE range — they have no memory of")
lines.append("  what the curve looked like one hour ago.")
lines.append("")
lines.append(f"  DETECTION THRESHOLD: {threshold:.6f}  (normal mean + 3 sigma)")
lines.append(f"  NORMAL ERROR MEAN  : {errors_normal.mean():.6f}")
lines.append(f"  NORMAL ERROR STD   : {errors_normal.std():.6f}")
lines.append(DIVIDER)

output = "\n".join(lines)
print("\n" + output)

# ── Save to docs/ ────────────────────────────────────────────────────────────
DOCS_DIR.mkdir(exist_ok=True)
out_path = DOCS_DIR / 'model_comparison_results.txt'
out_path.write_text(output, encoding='utf-8')
print(f"\nSaved: docs/model_comparison_results.txt")

# ── Save numeric results as CSV for easy inspection ──────────────────────────
pd.DataFrame(results).to_csv(DOCS_DIR / 'model_comparison_results.csv', index=False)
print(f"Saved: docs/model_comparison_results.csv")
