"""
Week 1 Day 2 — Three anomaly detection models for spacecraft battery fault.

All models are trained on NORMAL data only.  No fault labels are ever used
during training — this mirrors the real-world constraint that you only have
examples of a healthy spacecraft.

Run from project root:
    python models/train_models.py
"""

# Must be set before any keras import so it selects the PyTorch compute engine.
# TensorFlow doesn't support Python 3.14; Keras 3.x is backend-agnostic.
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
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score

SENSORS        = ['cabin_temperature', 'oxygen_percentage', 'battery_percentage', 'co2_percentage']
FAULT_START    = 4000          # minute fault begins
FAULT_END      = 4120          # minute of full failure  (FAULT_START + 120)
WINDOW_SIZE    = 60            # 1-hour lookback for LSTM
BATT_THRESHOLD = 15.0          # % — simple rule trigger
MODELS_DIR     = ROOT / 'models'


# ══════════════════════════════════════════════════════════════════════ Step 1
def load_data():
    print("=" * 65)
    print("Step 1  —  Load telemetry data")
    print("=" * 65)

    normal_df = pd.read_csv(ROOT / 'data' / 'normal_telemetry.csv', parse_dates=['timestamp'])
    fault_df  = pd.read_csv(ROOT / 'data' / 'fault_telemetry.csv',  parse_dates=['timestamp'])

    print(f"  Normal : {len(normal_df):,} rows  |  minutes 0 - {normal_df['minute'].max()}")
    print(f"  Fault  : {len(fault_df):,}  rows  |  slow_drift @ min {FAULT_START},"
          f"  full failure @ min {FAULT_END}")

    # Ground truth for precision: 1 = the fault has started
    fault_labels = (fault_df['minute'].values >= FAULT_START).astype(int)
    print(f"  Anomalous minutes : {fault_labels.sum():,} / {len(fault_labels):,}"
          f"  ({fault_labels.mean() * 100:.1f}% of dataset)")

    return normal_df, fault_df, fault_labels


# ══════════════════════════════════════════════════════════════════════ Step 2
def run_threshold(fault_df, fault_labels):
    print("\n" + "=" * 65)
    print("Step 2  —  Baseline rule  (battery < 15%)")
    print("=" * 65)

    flags = (fault_df['battery_percentage'].values < BATT_THRESHOLD).astype(int)

    in_fault_window = (fault_df['minute'].values >= FAULT_START) & (flags == 1)
    first_detection = int(fault_df['minute'].values[in_fault_window].min()) \
                      if in_fault_window.any() else None

    mbf  = (FAULT_END - first_detection) if first_detection else None
    prec = precision_score(fault_labels, flags, zero_division=0)

    if first_detection:
        print(f"  First detection  :  minute {first_detection}")
        print(f"  Before failure   :  {mbf} min")
    else:
        print("  NEVER TRIGGERED in drift window")
    print(f"  Precision        :  {prec:.3f}")

    return {
        'name': 'Threshold  (battery < 15%)',
        'first_detection_minute': first_detection,
        'minutes_before_failure': mbf,
        'precision': prec,
        'flags': flags,
    }


# ══════════════════════════════════════════════════════════════════════ Step 3
def run_isolation_forest(normal_df, fault_df, fault_labels):
    print("\n" + "=" * 65)
    print("Step 3  —  Isolation Forest  (scikit-learn)")
    print("=" * 65)

    # Fit scaler on NORMAL data only — fault statistics must never leak in
    scaler   = StandardScaler()
    X_normal = scaler.fit_transform(normal_df[SENSORS].values.astype(np.float32))
    X_fault  = scaler.transform(fault_df[SENSORS].values.astype(np.float32))

    print("  Training on 14,400 normal readings ...")
    iforest = IsolationForest(
        n_estimators=200,    # more trees = more stable isolation boundaries
        contamination=0.01,  # expect ~1% of training points to look unusual
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_normal)

    raw     = iforest.predict(X_fault)              # -1 anomaly, +1 normal
    flags   = (raw == -1).astype(int)

    in_fault_window = (fault_df['minute'].values >= FAULT_START) & (flags == 1)
    first_detection = int(fault_df['minute'].values[in_fault_window].min()) \
                      if in_fault_window.any() else None

    mbf  = (FAULT_END - first_detection) if first_detection else None
    prec = precision_score(fault_labels, flags, zero_division=0)

    if first_detection:
        print(f"  First detection  :  minute {first_detection}")
        print(f"  Before failure   :  {mbf} min")
    else:
        print("  Did not flag fault in drift window")
    print(f"  Precision        :  {prec:.3f}")
    print(f"  Total flags      :  {flags.sum():,}")

    joblib.dump(iforest, MODELS_DIR / 'isolation_forest.pkl')
    joblib.dump(scaler,  MODELS_DIR / 'if_scaler.pkl')
    print("  Saved: models/isolation_forest.pkl")

    return {
        'name': 'Isolation Forest',
        'first_detection_minute': first_detection,
        'minutes_before_failure': mbf,
        'precision': prec,
        'flags': flags,
        'scores': -iforest.score_samples(X_fault),
    }


# ══════════════════════════════════════════════════════════════════════ Step 4

def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Build (n_windows, window_size, n_features) array via fancy indexing.
    Each row is one contiguous slice of `window_size` timesteps.
    No Python loops — pure vectorised numpy.
    """
    n       = len(data) - window_size + 1
    indices = np.arange(window_size)[None, :] + np.arange(n)[:, None]
    return data[indices]


def build_lstm_autoencoder(window_size: int, n_features: int):
    """
    Encoder-Decoder LSTM that compresses a 60-min sequence to 32 numbers,
    then reconstructs the original sequence from those 32 numbers.

    Why not a plain dense autoencoder?
        Dense treats every timestep as independent.  An LSTM's hidden state
        carries information forward in time, so it learns that 'battery at
        minute T should be ~X given what happened at T-1 ... T-59.'  That
        temporal memory is exactly what lets it catch a slow drift early,
        before individual values leave the normal absolute range.

    Tanh activations inside LSTM:
        LSTM gates already use tanh/sigmoid internally.  Matching the output
        activation keeps the gradient flow healthy and avoids exploding values.
    """
    import keras
    from keras import layers

    inp     = keras.Input(shape=(window_size, n_features), name='input')
    x       = layers.LSTM(64, activation='tanh', return_sequences=False, name='enc_lstm')(inp)
    encoded = layers.Dense(32, activation='relu', name='bottleneck')(x)
    x       = layers.RepeatVector(window_size, name='repeat')(encoded)
    x       = layers.LSTM(64, activation='tanh', return_sequences=True,  name='dec_lstm')(x)
    out     = layers.TimeDistributed(layers.Dense(n_features), name='output')(x)

    model = keras.Model(inp, out, name='lstm_autoencoder')
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    return model


def run_lstm_autoencoder(normal_df, fault_df, fault_labels):
    print("\n" + "=" * 65)
    print("Step 4  —  LSTM Autoencoder  (Keras 3 / PyTorch backend)")
    print("=" * 65)

    try:
        import keras
        print(f"  Keras {keras.__version__}  |  backend: {keras.backend.backend()}")
    except ImportError as err:
        print(f"  ERROR: {err}")
        return None

    keras.utils.set_random_seed(42)

    scaler          = StandardScaler()
    X_normal_scaled = scaler.fit_transform(normal_df[SENSORS].values).astype(np.float32)
    X_fault_scaled  = scaler.transform(fault_df[SENSORS].values).astype(np.float32)

    print(f"  Building {WINDOW_SIZE}-min sliding windows ...")
    W_normal = create_windows(X_normal_scaled, WINDOW_SIZE)
    W_fault  = create_windows(X_fault_scaled,  WINDOW_SIZE)
    print(f"    Normal : {W_normal.shape}   Fault : {W_fault.shape}")

    model = build_lstm_autoencoder(WINDOW_SIZE, len(SENSORS))
    model.summary()

    print("\n  Training ...")
    history = model.fit(
        W_normal, W_normal,        # autoencoder: target == input
        epochs=50,
        batch_size=64,
        validation_split=0.1,      # last 10% of normal windows = val set
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5,
                restore_best_weights=True, verbose=0,
            )
        ],
        verbose=1,
    )
    print(f"  Converged in {len(history.history['loss'])} epochs")

    # ── Calibrate threshold on NORMAL data ──────────────────────────────
    recon_n       = model.predict(W_normal, batch_size=256, verbose=0)
    errors_normal = np.mean((W_normal - recon_n) ** 2, axis=(1, 2))
    threshold     = errors_normal.mean() + 3 * errors_normal.std()
    print(f"  Normal error  :  mean={errors_normal.mean():.6f}  std={errors_normal.std():.6f}")
    print(f"  Threshold     :  {threshold:.6f}  (mean + 3 sigma)")

    # ── Score fault dataset ──────────────────────────────────────────────
    recon_f      = model.predict(W_fault, batch_size=256, verbose=0)
    errors_fault = np.mean((W_fault - recon_f) ** 2, axis=(1, 2))

    # Assign each window's score to its LAST minute.
    # Real-time semantics: the decision is only available once the full
    # 60-minute window has been observed.
    pad    = np.full(WINDOW_SIZE - 1, np.nan)          # no decision for min 0-58
    errors = np.concatenate([pad, errors_fault])        # length = 14,400

    flags_raw = (errors_fault > threshold).astype(int)
    flags     = np.concatenate([np.zeros(WINDOW_SIZE - 1, dtype=int), flags_raw])

    in_fault_window = (fault_df['minute'].values >= FAULT_START) & (flags == 1)
    first_detection = int(fault_df['minute'].values[in_fault_window].min()) \
                      if in_fault_window.any() else None

    mbf  = (FAULT_END - first_detection) if first_detection else None
    prec = precision_score(fault_labels, flags, zero_division=0)

    if first_detection:
        print(f"  First detection  :  minute {first_detection}")
        print(f"  Before failure   :  {mbf} min")
    else:
        print("  Did not detect fault in drift window")
    print(f"  Precision        :  {prec:.3f}")

    try:
        model.save(str(MODELS_DIR / 'lstm_autoencoder.keras'))
        joblib.dump(scaler, MODELS_DIR / 'lstm_scaler.pkl')
        print("  Saved: models/lstm_autoencoder.keras")
    except Exception as exc:
        print(f"  Model save skipped  ({exc})")

    return {
        'name': 'LSTM Autoencoder',
        'first_detection_minute': first_detection,
        'minutes_before_failure': mbf,
        'precision': prec,
        'flags': flags,
        'errors': errors,
        'threshold': threshold,
        'errors_normal': errors_normal,
    }


# ══════════════════════════════════════════════════════════════════════ Step 5
def print_comparison_table(results):
    print("\n" + "=" * 65)
    print("Step 5  —  Model Comparison")
    print("=" * 65)

    col = f"\n  {'Model':<32}  {'First Detection':>15}  {'Before Failure':>15}  {'Precision':>10}"
    print(col)
    print("  " + "-" * 76)

    for r in results:
        fd  = f"min {r['first_detection_minute']}" if r['first_detection_minute'] else "NOT DETECTED"
        mbf = f"{r['minutes_before_failure']} min"  if r['minutes_before_failure'] else "N/A"
        print(f"  {r['name']:<32}  {fd:>15}  {mbf:>15}  {r['precision']:>10.3f}")

    lstm = next((r for r in results if 'LSTM' in r['name']), None)
    if lstm and lstm['first_detection_minute']:
        gap = lstm['minutes_before_failure']
        print(f"\n  The LSTM gives {gap} minutes of warning before catastrophic failure.")
        print(f"  That is the crew's intervention window.")


# ══════════════════════════════════════════════════════════════════════ Step 6
def plot_results(lstm_result, fault_df, save_path: Path):
    if lstm_result is None:
        print("LSTM unavailable — skipping plot.")
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={'height_ratios': [1, 2.2], 'hspace': 0.06},
    )
    fig.patch.set_facecolor('#0d1117')

    minutes   = fault_df['minute'].values
    errors    = lstm_result['errors']
    threshold = lstm_result['threshold']
    fd        = lstm_result['first_detection_minute']

    def style(ax):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.5)

    # ── Top: battery reading ─────────────────────────────────────────────
    style(ax_top)
    ax_top.plot(minutes, fault_df['battery_percentage'].values,
                color='#32CD32', linewidth=0.7, label='Battery %')
    ax_top.axhline(BATT_THRESHOLD, color='#FFD700', linewidth=1.0, linestyle='--',
                   alpha=0.75, label=f'Rule threshold  ({BATT_THRESHOLD}%)')
    ax_top.axvline(FAULT_START, color='#ff4444', linewidth=1.5, linestyle='--', alpha=0.9)
    ax_top.axvline(FAULT_END,   color='#cc0000', linewidth=1.5, linestyle='-',  alpha=0.9)
    if fd:
        ax_top.axvline(fd, color='#00FF7F', linewidth=2.0, alpha=0.95)
    ax_top.axvspan(FAULT_START, FAULT_END, alpha=0.07, color='red')
    ax_top.set_ylabel('Battery (%)', color='#c9d1d9', fontsize=10)
    ax_top.set_xticklabels([])
    ax_top.set_title(
        'LSTM Autoencoder  —  Anomaly Detection  —  Battery slow_drift Fault',
        color='white', fontsize=13, fontweight='bold', pad=10,
    )
    ax_top.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # ── Bottom: reconstruction error ─────────────────────────────────────
    style(ax_bot)
    errs_safe = np.where(np.isnan(errors), 0.0, errors)

    ax_bot.plot(minutes, errors, color='#00BFFF', linewidth=0.7, alpha=0.9,
                label='Reconstruction error (MSE per window)')
    ax_bot.axhline(threshold, color='#FFD700', linewidth=1.5, linestyle='--',
                   label=f'Detection threshold  ({threshold:.5f}  =  mean + 3 sigma)')
    ax_bot.axvline(FAULT_START, color='#ff4444', linewidth=1.5, linestyle='--', alpha=0.9,
                   label=f'Fault begins  (min {FAULT_START})')
    ax_bot.axvline(FAULT_END,   color='#cc0000', linewidth=1.5, linestyle='-',  alpha=0.9,
                   label=f'Full failure  (min {FAULT_END})')
    ax_bot.axvspan(FAULT_START, FAULT_END, alpha=0.07, color='red')

    ax_bot.fill_between(
        minutes, threshold, errs_safe,
        where=errs_safe > threshold,
        alpha=0.22, color='orange', label='Flagged zone',
    )

    if fd:
        fd_err = float(errors[fd]) if not np.isnan(errors[fd]) else threshold * 1.1
        ax_bot.axvline(fd, color='#00FF7F', linewidth=2.0, alpha=0.95,
                       label=f'FIRST DETECTION  (min {fd}  —  {FAULT_END - fd} min before failure)')
        ax_bot.scatter([fd], [fd_err], color='#00FF7F', s=90, zorder=10)
        ax_bot.annotate(
            f"DETECTED\nmin {fd}\n{FAULT_END - fd} min\nbefore failure",
            xy=(fd, fd_err), xytext=(25, 35), textcoords='offset points',
            color='#00FF7F', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00FF7F', lw=1.2),
        )

    ax_bot.set_xlabel('Mission time (minutes)', color='#c9d1d9', fontsize=10)
    ax_bot.set_ylabel('Reconstruction Error (MSE)', color='#c9d1d9', fontsize=10)
    ax_bot.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#c9d1d9', loc='upper left')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nPlot saved: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════ Main
def main():
    normal_df, fault_df, fault_labels = load_data()

    threshold_result = run_threshold(fault_df, fault_labels)
    iforest_result   = run_isolation_forest(normal_df, fault_df, fault_labels)
    lstm_result      = run_lstm_autoencoder(normal_df, fault_df, fault_labels)

    all_results = [threshold_result, iforest_result]
    if lstm_result:
        all_results.append(lstm_result)

    print_comparison_table(all_results)

    plot_path = ROOT / 'day2_anomaly_detection_results.png'
    plot_results(lstm_result, fault_df, plot_path)

    print("\nDay 2 complete. Study the detection plot and share your feedback.")


if __name__ == '__main__':
    main()
