"""
Week 1 Day 1 — generate normal and fault telemetry, save CSVs, show side-by-side plot.

Run from the spacecraft-monitor/ root:
    python data/generate_datasets.py
"""

from pathlib import Path
import sys

# Ensure project root is on the path so 'from data.generator import ...' works
# regardless of which directory the script is launched from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from data.generator import SpacecraftTelemetryGenerator


FAULT_START    = 4000         # minute where the battery fault begins
DRIFT_DURATION = 120          # 2-hour ramp (matches inject_fault hard-code)
DRIFT_END      = FAULT_START + DRIFT_DURATION

# (column key, y-axis label, line colour)
SENSORS = [
    ('cabin_temperature',  'Cabin Temp (°F)',  '#FF8C00'),
    ('oxygen_percentage',  'Oxygen (%)',        '#00BFFF'),
    ('battery_percentage', 'Battery (%)',       '#32CD32'),
    ('co2_percentage',     'CO₂ (%)',           '#DC143C'),
]


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

def plot_comparison(normal_df, fault_df, save_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(20, 12),
        sharex=True,
        gridspec_kw={'hspace': 0.40, 'wspace': 0.10},
    )
    fig.patch.set_facecolor('#0d1117')   # dark GitHub-style background

    days = normal_df['minute'].values / 1440.0  # convert minutes → days for readability

    col_meta = [
        ('NORMAL  —  No Faults',                  '#1f6feb', normal_df),
        ('FAULT  —  Battery slow_drift @ min 4000', '#da3633', fault_df),
    ]

    for col, (col_title, col_color, df) in enumerate(col_meta):
        for row, (sensor, ylabel, line_color) in enumerate(SENSORS):
            ax = axes[row][col]
            ax.set_facecolor('#161b22')
            ax.plot(days, df[sensor].values, color=line_color, linewidth=0.6, alpha=0.85)
            ax.set_ylabel(ylabel, color='#c9d1d9', fontsize=9)
            ax.tick_params(colors='#8b949e', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')
            ax.grid(True, color='#21262d', linewidth=0.5)

            if row == 0:
                ax.set_title(
                    col_title, fontsize=11, fontweight='bold', color='white', pad=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=col_color, alpha=0.85),
                )

            # Fault markers on right column only
            if col == 1:
                fault_day_start = FAULT_START / 1440
                fault_day_end   = DRIFT_END   / 1440
                ax.axvline(fault_day_start, color='#ff4444', linewidth=1.3,
                           linestyle='--', alpha=0.95, zorder=5)
                ax.axvline(fault_day_end,   color='#ff8800', linewidth=1.1,
                           linestyle=':',  alpha=0.80, zorder=5)
                ax.axvspan(fault_day_start, fault_day_end,
                           alpha=0.07, color='red', zorder=3)

    # Shared x-axis label (bottom row only)
    for ax in axes[3]:
        ax.set_xlabel('Mission time (days)', color='#c9d1d9', fontsize=9)

    # Legend — placed in the battery-fault subplot (row 2, col 1)
    legend_handles = [
        mlines.Line2D([], [], color='#ff4444', linestyle='--', linewidth=1.3,
                      label=f'Fault begins  (day {FAULT_START/1440:.2f} / min {FAULT_START})'),
        mlines.Line2D([], [], color='#ff8800', linestyle=':', linewidth=1.1,
                      label=f'Full failure  (day {DRIFT_END/1440:.2f} / min {DRIFT_END})'),
        mpatches.Patch(facecolor='red', alpha=0.25,
                       label='2-hour drift window  (ML must catch this)'),
    ]
    axes[2][1].legend(
        handles=legend_handles, fontsize=8,
        facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9',
        loc='lower right',
    )

    fig.suptitle(
        'Orion Spacecraft Telemetry — Week 1 Day 1 Validation',
        fontsize=14, fontweight='bold', color='white', y=0.997,
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Plot saved: {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    gen = SpacecraftTelemetryGenerator(seed=42)

    print("Generating normal telemetry (14,400 readings = 10 days @ 1 Hz)...")
    normal_df = gen.generate()

    print(f"Injecting slow_drift battery fault at minute {FAULT_START}...")
    fault_df = gen.inject_fault(
        normal_df,
        sensor_name='battery_percentage',
        fault_start_time=FAULT_START,
        fault_type='slow_drift',
    )

    data_dir = ROOT / 'data'
    normal_df.to_csv(data_dir / 'normal_telemetry.csv', index=False)
    fault_df.to_csv(data_dir  / 'fault_telemetry.csv',  index=False)
    print(f"  Saved: data/normal_telemetry.csv  ({len(normal_df):,} rows)")
    print(f"  Saved: data/fault_telemetry.csv   ({len(fault_df):,} rows)")

    # ---- sanity numbers ------------------------------------------------
    def batt(df, minute):
        return df.loc[df['minute'] == minute, 'battery_percentage'].values[0]

    n_start  = batt(normal_df, FAULT_START)
    f_start  = batt(fault_df,  FAULT_START)
    n_mid    = batt(normal_df, FAULT_START + 60)
    f_mid    = batt(fault_df,  FAULT_START + 60)
    f_full   = batt(fault_df,  DRIFT_END)

    print("\nSanity check — battery_percentage:")
    print(f"  min {FAULT_START:5d}  normal={n_start:6.2f}%   fault={f_start:6.2f}%  (delta={f_start-n_start:+.2f}%)")
    print(f"  min {FAULT_START+60:5d}  normal={n_mid:6.2f}%   fault={f_mid:6.2f}%  (delta={f_mid-n_mid:+.2f}%  <- midpoint of drift)")
    print(f"  min {DRIFT_END:5d}  fault={f_full:6.2f}%  (full failure reached)")

    print("\nPlotting comparison...")
    plot_path = ROOT / 'day1_telemetry_comparison.png'
    plot_comparison(normal_df, fault_df, plot_path)

    print("\nAll done! Study the graph, then share your feedback.")


if __name__ == '__main__':
    main()
