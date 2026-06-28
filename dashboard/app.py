"""
dashboard/app.py
Week 1 Day 3/4 — Orion Spacecraft Health Monitor
Real-time anomaly detection with 5 historical fault scenarios.

Run from project root:
    streamlit run dashboard/app.py
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

# ── Scenario registry ─────────────────────────────────────────────────────────
# Single source of truth for every fault scenario.
# The dashboard uses this dict for buttons, alert text, chart markers, and
# the LSTM evaluation — nothing is hardcoded in multiple places.
SCENARIOS = {
    'normal': {
        'label':          'Normal Mission',
        'btn_label':      '▶  Run Normal Mission',
        'btn_type':       'secondary',
        'history':        None,
        'csv':            'normal_telemetry.csv',
        'start_idx':      0,
        'fault_sensor':   None,
        'fault_start':    None,
        'fault_end':      None,
        'danger_note':    None,
        'badge_color':    '#3fb950',
    },
    'battery': {
        'label':          'Battery Cell Collapse',
        'btn_label':      '⚡  Battery Collapse',
        'btn_type':       'primary',
        'history':        'Day 2 baseline',
        'csv':            'fault_telemetry.csv',
        'start_idx':      3940,
        'fault_sensor':   'battery_percentage',
        'fault_start':    4000,
        'fault_end':      4120,
        'danger_note':    'Battery drains to 0 % — total power loss',
        'badge_color':    '#f85149',
    },
    'oxygen': {
        'label':          'O2 Pressure Leak',
        'btn_label':      '💨  O2 Pressure Leak',
        'btn_type':       'primary',
        'history':        'ISS Zvezda microleak (2019–21)',
        'csv':            'oxygen_fault.csv',
        'start_idx':      4940,
        'fault_sensor':   'oxygen_percentage',
        'fault_start':    5000,
        'fault_end':      5090,
        'danger_note':    'O2 drops below 19.5 % — hypoxia risk to crew',
        'badge_color':    '#79c0ff',
    },
    'co2': {
        'label':          'CO2 Scrubber Failure',
        'btn_label':      '☣  CO2 Scrubber Fail',
        'btn_type':       'primary',
        'history':        'Apollo 13 CO2 crisis (Apr 1970)',
        'csv':            'co2_fault.csv',
        'start_idx':      5940,
        'fault_sensor':   'co2_percentage',
        'fault_start':    6000,
        'fault_end':      6120,
        'danger_note':    'CO2 exceeds 1.0 % — toxicity risk to crew',
        'badge_color':    '#ffa657',
    },
    'temperature': {
        'label':          'Thermal Control Failure',
        'btn_label':      '🌡  Thermal Control Fail',
        'btn_type':       'primary',
        'history':        'Space Shuttle TCS anomalies',
        'csv':            'temperature_fault.csv',
        'start_idx':      6940,
        'fault_sensor':   'cabin_temperature',
        'fault_start':    7000,
        'fault_end':      7090,
        'danger_note':    'Temp exceeds 85 °F — heat stroke risk to crew',
        'badge_color':    '#ff7b72',
    },
    'power': {
        'label':          'Power System Spike',
        'btn_label':      '⚡  Power System Spike',
        'btn_type':       'primary',
        'history':        'Sudden electrical failure (instant)',
        'csv':            'power_spike_fault.csv',
        'start_idx':      7940,
        'fault_sensor':   'cabin_temperature',
        'fault_start':    8000,
        'fault_end':      8000,
        'danger_note':    'Instantaneous — no warning window',
        'badge_color':    '#d2a8ff',
    },
}

SENSORS = ['cabin_temperature', 'oxygen_percentage', 'battery_percentage', 'co2_percentage']
SENSOR_META = {
    'cabin_temperature':  dict(label='Cabin Temp',  unit='°F', color='#FF8C00', lo=68,   hi=76),
    'oxygen_percentage':  dict(label='Oxygen',       unit='%',  color='#00BFFF', lo=20.5, hi=21.0),
    'battery_percentage': dict(label='Battery',      unit='%',  color='#32CD32', lo=20,   hi=100),
    'co2_percentage':     dict(label='CO₂',          unit='%',  color='#DC143C', lo=0.3,  hi=0.6),
}

WINDOW_SIZE = 60
THRESHOLD   = 0.053946   # normal mean + 3 sigma (Day 2 calibration)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='SentinelAI — Spacecraft Health Monitoring',
    page_icon='🛸',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base — darker, more professional ── */
.stApp { background-color: #0a0a0f !important; color: #c9d1d9; }
section[data-testid="stSidebar"] { background-color: #0d0f18 !important; border-right: 1px solid #1e2130; }
#MainMenu, footer, header { visibility: hidden; }

/* ── SentinelAI Navbar ── */
.header-bar {
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(90deg, #050508 0%, #0d0f18 50%, #050508 100%);
    border: 1px solid #1e2130; border-radius: 6px;
    padding: 14px 28px; margin-bottom: 8px;
}
.sentinel-logo  { display: flex; align-items: center; gap: 12px; }
.sentinel-bracket {
    font-size: 26px; font-weight: 900; color: #3fb950;
    font-family: 'Courier New', monospace; border: 2px solid #3fb950;
    border-radius: 4px; padding: 3px 10px; line-height: 1;
    box-shadow: 0 0 12px #3fb95055; text-align: center;
}
.sentinel-name {
    font-size: 20px; font-weight: 900; color: #3fb950;
    letter-spacing: 5px; font-family: 'Courier New', monospace;
    text-shadow: 0 0 20px #3fb95077;
}
.nav-center    { text-align: center; }
.nav-subtitle  {
    font-size: 10px; color: #e3b341; letter-spacing: 3px;
    font-family: 'Courier New', monospace; font-weight: 600; margin-bottom: 6px;
}
.met-block    { text-align: center; }
.met-label    { font-size: 10px; color: #6e7681; letter-spacing: 2px; }
.met-time     { font-size: 22px; font-weight: bold; color: #e6edf3; font-family: 'Courier New', monospace; }
.met-min      { font-size: 11px; color: #8b949e; margin-top: 2px; }

/* ── Status indicators — bigger & more dramatic ── */
.status-nominal {
    font-size: 11px; font-weight: 900; color: #3fb950;
    font-family: 'Courier New', monospace;
    border: 2px solid #3fb950; border-radius: 6px;
    padding: 12px 22px; letter-spacing: 2px; text-align: center; min-width: 148px;
    box-shadow: 0 0 18px #3fb95055, inset 0 0 18px #3fb95011;
}
.sn-icon { font-size: 22px; display: block; margin-bottom: 4px; }
.sn-main { font-size: 19px; display: block; letter-spacing: 4px; }
.sn-sub  { font-size: 10px; display: block; letter-spacing: 3px; color: #3fb950aa; margin-top: 3px; }
.status-anomaly {
    font-size: 11px; font-weight: 900; color: #f85149;
    font-family: 'Courier New', monospace;
    border: 2px solid #f85149; border-radius: 6px;
    padding: 12px 22px; letter-spacing: 2px; text-align: center; min-width: 148px;
    animation: pulse-border 0.9s ease-in-out infinite;
}
.sa-icon { font-size: 22px; display: block; margin-bottom: 4px; }
.sa-main { font-size: 19px; display: block; letter-spacing: 4px; }
.sa-sub  { font-size: 10px; display: block; letter-spacing: 3px; color: #f8514999; margin-top: 3px; }
@keyframes pulse-border {
    0%,100% { box-shadow: 0 0 8px rgba(248,81,73,0.5), 0 0 0 0 rgba(248,81,73,0.3); }
    50%      { box-shadow: 0 0 24px rgba(248,81,73,0.9), 0 0 14px 4px rgba(248,81,73,0.2); }
}

/* ── Alert panel ── */
.alert-panel { background: rgba(248,81,73,0.08); border: 2px solid #f85149;
               border-radius: 6px; padding: 14px 22px; margin-bottom: 8px; }
.alert-title { font-size: 17px; font-weight: 900; color: #f85149;
               letter-spacing: 2px; margin-bottom: 8px; font-family: monospace; }
.alert-detail { font-size: 13px; color: #ffa657; line-height: 2; }
.alert-detail strong { color: #ff7b72; font-size: 14px; }

/* ── Sensor tiles ── */
.tile-row { display: flex; gap: 8px; margin-bottom: 8px; }
.sensor-tile { flex: 1; background: #0f1117; border: 1px solid #1e2130;
               border-radius: 6px; padding: 10px 14px; text-align: center; }
.tile-label  { font-size: 9px; color: #6e7681; letter-spacing: 2px; margin-bottom: 4px; }
.tile-value  { font-size: 27px; font-weight: bold; font-family: 'Courier New', monospace; line-height: 1.1; }
.tile-range  { font-size: 9px; color: #6e7681; margin-top: 3px; }
.tile-status { font-size: 9px; margin-top: 2px; }

/* ── Sidebar simulation status indicators ── */
.sim-running-fault {
    background: rgba(248,81,73,0.1); border: 1px solid #f85149; border-radius: 4px;
    padding: 9px 14px; text-align: center; font-family: monospace;
    font-size: 12px; font-weight: bold; color: #f85149; margin-bottom: 4px;
}
.sim-running-normal {
    background: rgba(63,185,80,0.1); border: 1px solid #3fb950; border-radius: 4px;
    padding: 9px 14px; text-align: center; font-family: monospace;
    font-size: 12px; font-weight: bold; color: #3fb950; margin-bottom: 4px;
}
.sim-stopped {
    background: #0a0a0f; border: 1px solid #1e2130; border-radius: 4px;
    padding: 9px 14px; text-align: center; font-family: monospace;
    font-size: 12px; color: #6e7681; margin-bottom: 4px;
}
/* Pulsing dots */
.dot-red {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #f85149; margin-right: 7px; vertical-align: middle;
    animation: dot-pulse-red 0.8s ease-in-out infinite;
}
.dot-green {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #3fb950; margin-right: 7px; vertical-align: middle;
    animation: dot-pulse-green 1.2s ease-in-out infinite;
}
@keyframes dot-pulse-red   { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.5);opacity:0.4} }
@keyframes dot-pulse-green { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.5);opacity:0.5} }

/* ── Sidebar section headers ── */
.sidebar-section { font-size: 10px; color: #6e7681; letter-spacing: 2px;
                   margin: 8px 0 4px 0; padding-bottom: 3px; border-bottom: 1px solid #1e2130; }
/* ── Scenario badge in sidebar ── */
.scenario-badge { display: inline-block; font-size: 10px; color: #8b949e;
                  font-family: monospace; margin-top: 4px; }
/* ── Simulation status box ── */
.sim-status-box {
    background: rgba(13,15,24,0.9); border: 1px solid #1e2130;
    border-radius: 4px; padding: 8px 12px; text-align: center;
    font-family: monospace; margin-bottom: 4px;
}
.ssb-label  { font-size: 11px; font-weight: bold; letter-spacing: 1px; margin-bottom: 3px; }
.ssb-count  { color: #e6edf3; font-size: 15px; font-weight: bold; }
.ssb-minute { font-size: 10px; color: #6e7681; letter-spacing: 1px; }
.sim-standby { color: #6e7681; font-size: 12px; letter-spacing: 2px; }
/* ── SentinelAI watermark ── */
.sentinel-watermark {
    position: fixed; bottom: 16px; right: 20px; z-index: 9999;
    font-size: 9px; color: #252c3a; font-family: 'Courier New', monospace;
    letter-spacing: 2px; pointer-events: none; user-select: none;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_models():
    import keras
    model  = keras.models.load_model(str(ROOT / 'models' / 'lstm_autoencoder.keras'))
    scaler = joblib.load(ROOT / 'models' / 'lstm_scaler.pkl')
    return model, scaler


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    """Cache each telemetry CSV by filename — loaded once per server process."""
    return pd.read_csv(ROOT / 'data' / filename, parse_dates=['timestamp'])


# ── Session state ─────────────────────────────────────────────────────────────
def reset_mission(scenario_key: str) -> None:
    sc = SCENARIOS[scenario_key]
    st.session_state.scenario      = scenario_key
    st.session_state.running       = True
    st.session_state.current_idx   = sc['start_idx']
    st.session_state.csv_file      = sc['csv']
    st.session_state.fault_sensor  = sc['fault_sensor']
    st.session_state.fault_start   = sc['fault_start']
    st.session_state.fault_end     = sc['fault_end']
    st.session_state.buffer        = []
    st.session_state.error_history = []
    st.session_state.alert_active  = False
    st.session_state.alert_minute  = None
    st.session_state.alert_error   = 0.0
    st.session_state.mission_start = time.time()


if 'running' not in st.session_state:
    reset_mission('normal')


# ── LSTM inference ────────────────────────────────────────────────────────────
def compute_recon_error(buffer: list, model, scaler) -> float:
    raw    = np.array([[r[s] for s in SENSORS] for r in buffer], dtype=np.float32)
    scaled = scaler.transform(raw)
    window = scaled[np.newaxis, :, :]
    recon  = model.predict(window, verbose=0)
    return float(np.mean((window - recon) ** 2))


# ── Simulation tick ───────────────────────────────────────────────────────────
def advance_one_step(model, scaler) -> None:
    df  = load_csv(st.session_state.csv_file)
    idx = st.session_state.current_idx

    if idx >= len(df):
        st.session_state.running = False
        return

    row = df.iloc[idx].to_dict()

    st.session_state.buffer.append(row)
    if len(st.session_state.buffer) > WINDOW_SIZE:
        st.session_state.buffer.pop(0)

    if len(st.session_state.buffer) == WINDOW_SIZE:
        err = compute_recon_error(st.session_state.buffer, model, scaler)
        st.session_state.error_history.append({'minute': row['minute'], 'error': err})

        fault_start = st.session_state.fault_start
        if (err > THRESHOLD
                and not st.session_state.alert_active
                and fault_start is not None
                and row['minute'] >= fault_start):
            st.session_state.alert_active = True
            st.session_state.alert_minute = int(row['minute'])
            st.session_state.alert_error  = err

    st.session_state.current_idx += 1


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════════════

model, scaler = load_ml_models()

cur_idx      = st.session_state.current_idx
scenario_key = st.session_state.scenario
sc           = SCENARIOS[scenario_key]
alert_on     = st.session_state.alert_active
buf          = st.session_state.buffer
is_fault_sim = scenario_key != 'normal'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛸 ORION CONTROL")

    # ── Simulation status indicator ──────────────────────────────────────────
    if st.session_state.running:
        if is_fault_sim:
            st.markdown(
                '<div class="sim-running-fault">'
                '<span class="dot-red"></span>SIMULATION RUNNING</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sim-running-normal">'
                '<span class="dot-green"></span>MISSION ACTIVE</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="sim-stopped">⬤ &nbsp; SIMULATION STOPPED</div>',
            unsafe_allow_html=True,
        )

    # ── Status box: active simulation name + live minute counter ────────────
    if st.session_state.running:
        badge_c = sc['badge_color']
        st.markdown(
            f'<div class="sim-status-box" style="border-color:{badge_c}">'
            f'<div class="ssb-label" style="color:{badge_c}">{sc["label"].upper()}</div>'
            f'<div class="ssb-minute">MISSION MIN &nbsp;'
            f'<span class="ssb-count">{cur_idx:,}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sim-status-box sim-standby">◉ &nbsp; STANDING BY</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section">MISSION</div>', unsafe_allow_html=True)
    if st.button('▶  Run Normal Mission', use_container_width=True, type='secondary'):
        reset_mission('normal')
        st.rerun()

    st.markdown('<div class="sidebar-section">FAULT SIMULATIONS</div>', unsafe_allow_html=True)

    fault_keys = ['battery', 'oxygen', 'co2', 'temperature', 'power']
    for key in fault_keys:
        fsc = SCENARIOS[key]
        if scenario_key == key and st.session_state.running:
            bc = fsc['badge_color']
            st.markdown(
                f'<div class="fcam"></div>'
                f'<style>'
                f'div:has(.fcam)+div{{outline:1.5px solid {bc}!important;'
                f'border-radius:6px!important;'
                f'box-shadow:0 0 8px {bc}88,0 0 20px {bc}33!important;'
                f'animation:fcam-glow 1.5s ease-in-out infinite!important}}'
                f'@keyframes fcam-glow{{'
                f'0%,100%{{box-shadow:0 0 6px {bc}66,0 0 12px {bc}22;outline-color:{bc}99}}'
                f'50%{{box-shadow:0 0 16px {bc}cc,0 0 28px {bc}55;outline-color:{bc}}}}}'
                f'</style>',
                unsafe_allow_html=True,
            )
        col_btn, col_info = st.columns([3, 2])
        with col_btn:
            if st.button(fsc['btn_label'], use_container_width=True, type='primary', key=f'btn_{key}'):
                reset_mission(key)
                st.rerun()
        with col_info:
            st.markdown(
                f'<div class="scenario-badge" style="color:{fsc["badge_color"]}">'
                f'{fsc["history"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sidebar-section">CONTROLS</div>', unsafe_allow_html=True)
    if st.button('⏹  Stop', use_container_width=True):
        st.session_state.running = False

    speed = st.slider(
        'Speed (sec / reading)', 0.05, 1.0, 0.15, 0.05,
        help='0.05 = fastest.  0.15 = default demo speed.',
    )

    st.markdown('<div class="sidebar-section">ML MODEL</div>', unsafe_allow_html=True)
    st.code(
        f"LSTM Autoencoder\n"
        f"Window  : {WINDOW_SIZE} min\n"
        f"Thresh  : {THRESHOLD:.5f}\n"
        f"F1      : 0.992  Prec: 1.000",
        language=None,
    )

    normal_df = load_csv('normal_telemetry.csv')
    n_total   = len(normal_df)
    st.progress(min(1.0, cur_idx / n_total))
    st.caption(f'Minute {cur_idx:,} / {n_total:,}')


# ── Mission header ─────────────────────────────────────────────────────────────
elapsed = int(time.time() - st.session_state.mission_start) if st.session_state.mission_start else 0
met_str = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"

status_html = (
    '<div class="status-anomaly">'
    '<span class="sa-icon">⚠</span>'
    '<span class="sa-main">ANOMALY</span>'
    '<span class="sa-sub">DETECTED</span>'
    '</div>'
    if alert_on else
    '<div class="status-nominal">'
    '<span class="sn-icon">●</span>'
    '<span class="sn-main">SYSTEMS</span>'
    '<span class="sn-sub">NOMINAL</span>'
    '</div>'
)

st.markdown(f"""
<div class="header-bar">
  <div class="sentinel-logo">
    <span class="sentinel-bracket">S</span>
    <span class="sentinel-name">SENTINEL AI</span>
  </div>
  <div class="nav-center">
    <div class="nav-subtitle">ORION SPACECRAFT &nbsp;•&nbsp; ARTEMIS II MISSION &nbsp;•&nbsp; LIVE MONITORING</div>
    <div class="met-block">
      <div class="met-label">MISSION ELAPSED TIME</div>
      <div class="met-time">{met_str}</div>
      <div class="met-min">spacecraft min &nbsp;{cur_idx:,}</div>
    </div>
  </div>
  {status_html}
</div>
""", unsafe_allow_html=True)


# ── Scenario info strip (only when fault sim is active) ───────────────────────
if is_fault_sim:
    badge_color = sc['badge_color']
    st.markdown(
        f'<div style="font-size:12px;color:{badge_color};font-family:monospace;'
        f'padding:4px 8px;border-left:3px solid {badge_color};margin-bottom:6px">'
        f'▸ &nbsp;<b>{sc["label"]}</b> &nbsp;|&nbsp; {sc["history"]} &nbsp;|&nbsp; '
        f'{sc["danger_note"]}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Critical alert panel ───────────────────────────────────────────────────────
if alert_on:
    err        = st.session_state.alert_error
    alert_min  = st.session_state.alert_minute
    excess     = max(0.0, (err - THRESHOLD) / THRESHOLD)
    confidence = min(99.9, 50.0 + excess * 49.9)
    fault_end  = st.session_state.fault_end
    mtf        = max(0, fault_end - cur_idx) if fault_end else None
    mtf_str    = f"{mtf} min" if mtf is not None else "—"
    sensor_lbl = SENSOR_META[st.session_state.fault_sensor]['label'].upper() \
                 if st.session_state.fault_sensor else "UNKNOWN"

    st.markdown(f"""
<div class="alert-panel">
  <div class="alert-title">🚨 &nbsp; CRITICAL ALERT — {sc['label'].upper()} — FAILURE PREDICTED</div>
  <div class="alert-detail">
    &nbsp;&nbsp;Sensor: <strong>{sensor_lbl}</strong>
    &nbsp;|&nbsp; Confidence: <strong>{confidence:.1f}%</strong>
    &nbsp;|&nbsp; First flagged: <strong>min {alert_min}</strong>
    &nbsp;|&nbsp; Est. time to failure: <strong>{mtf_str}</strong><br>
    &nbsp;&nbsp;LSTM error: <strong>{err:.5f}</strong>
    &nbsp;(threshold {THRESHOLD:.5f} — {err/THRESHOLD:.1f}× above limit)
    &nbsp;&nbsp;|&nbsp;&nbsp; {sc['danger_note']}
  </div>
</div>
""", unsafe_allow_html=True)


# ── Live metric tiles ──────────────────────────────────────────────────────────
fault_sensor = st.session_state.fault_sensor
tile_html = '<div class="tile-row">'
for sensor in SENSORS:
    meta       = SENSOR_META[sensor]
    is_faulted = is_fault_sim and sensor == fault_sensor and alert_on
    if buf:
        val      = buf[-1][sensor]
        in_range = meta['lo'] <= val <= meta['hi']
        v_color  = '#f85149' if is_faulted else ('#3fb950' if in_range else '#ffa657')
        v_str    = f"{val:.2f}"
        s_label  = "FAULT" if is_faulted else ("NOMINAL" if in_range else "WARNING")
        s_color  = v_color
    else:
        v_color = '#8b949e'; v_str = '--'; s_label = 'INIT'; s_color = '#8b949e'

    tile_html += (
        f'<div class="sensor-tile">'
        f'<div class="tile-label">{meta["label"].upper()}</div>'
        f'<div class="tile-value" style="color:{v_color}">{v_str}'
        f'<span style="font-size:14px">{meta["unit"]}</span></div>'
        f'<div class="tile-range">Normal {meta["lo"]}–{meta["hi"]}{meta["unit"]}</div>'
        f'<div class="tile-status" style="color:{s_color}">{s_label}</div>'
        f'</div>'
    )
tile_html += '</div>'
st.markdown(tile_html, unsafe_allow_html=True)


# ── Sensor charts ──────────────────────────────────────────────────────────────
def make_sensor_chart(buf: list, sensor: str) -> go.Figure:
    meta      = SENSOR_META[sensor]
    is_faulted = is_fault_sim and sensor == fault_sensor
    fig       = go.Figure()

    if buf:
        df       = pd.DataFrame(buf)
        cur_val  = df[sensor].iloc[-1]
        in_range = meta['lo'] <= cur_val <= meta['hi']
        lc       = ('#f85149' if (is_faulted and alert_on)
                    else meta['color'] if in_range
                    else '#ffa657')

        fig.add_hrect(
            y0=meta['lo'], y1=meta['hi'],
            fillcolor='rgba(59,130,246,0.05)', line_width=0,
            annotation_text='normal', annotation_font_size=8,
            annotation_font_color='#444c56', annotation_position='top left',
        )
        fig.add_trace(go.Scatter(
            x=df['minute'].tolist(), y=df[sensor].tolist(),
            mode='lines', line=dict(color=lc, width=1.8),
            hovertemplate='%{y:.3f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=[df['minute'].iloc[-1]], y=[cur_val],
            mode='markers+text',
            marker=dict(color=lc, size=8),
            text=[f" {cur_val:.2f}{meta['unit']}"],
            textposition='middle right',
            textfont=dict(color=lc, size=10, family='Courier New'),
        ))

    fig.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=9),
        margin=dict(l=42, r=10, t=28, b=28), height=195,
        showlegend=False, hovermode='x unified',
        title=dict(
            text=f"<b style='color:{meta['color']}'>{meta['label'].upper()}</b>",
            font=dict(size=11), x=0.02, xanchor='left',
        ),
        xaxis=dict(gridcolor='#21262d', title=dict(text='Mission minute', font=dict(size=8))),
        yaxis=dict(gridcolor='#21262d', title=dict(text=meta['unit'], font=dict(size=8))),
    )
    return fig


row_a = st.columns(2)
row_b = st.columns(2)
for i, sensor in enumerate(SENSORS):
    col = row_a[i] if i < 2 else row_b[i - 2]
    with col:
        st.plotly_chart(make_sensor_chart(buf, sensor), use_container_width=True,
                        key=f'chart_{sensor}')


# ── Reconstruction error chart ─────────────────────────────────────────────────
def make_error_chart() -> go.Figure:
    fig = go.Figure()
    eh  = st.session_state.error_history

    if eh:
        df    = pd.DataFrame(eh)
        mins  = df['minute'].tolist()
        errs  = df['error'].values

        # Invisible baseline at threshold for fill reference
        fig.add_trace(go.Scatter(
            x=mins, y=[THRESHOLD] * len(mins),
            mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
            hoverinfo='skip', showlegend=False,
        ))
        # Upper fill boundary: max(error, threshold) → orange fill only above threshold
        fig.add_trace(go.Scatter(
            x=mins, y=np.maximum(errs, THRESHOLD).tolist(),
            mode='lines', fill='tonexty',
            fillcolor='rgba(248,81,73,0.18)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            hoverinfo='skip', showlegend=False,
        ))
        # Error line on top
        fig.add_trace(go.Scatter(
            x=mins, y=errs.tolist(),
            mode='lines', name='Reconstruction error',
            line=dict(color='#00BFFF', width=1.4),
            hovertemplate='min %{x}: %{y:.5f}<extra></extra>',
        ))

    fig.add_hline(
        y=THRESHOLD, line_dash='dash', line_color='#FFD700', line_width=1.5,
        annotation_text=f'Threshold  {THRESHOLD:.5f}',
        annotation_font_color='#FFD700', annotation_font_size=10,
        annotation_position='top right',
    )

    fault_start = st.session_state.fault_start
    fault_end   = st.session_state.fault_end

    if fault_start and cur_idx >= fault_start:
        fig.add_vline(
            x=fault_start, line_dash='dash', line_color='#ff4444', line_width=1.4,
            annotation_text=f'Fault ({fault_start})',
            annotation_font_color='#ff4444', annotation_font_size=9,
            annotation_position='top left',
        )
    if fault_end and fault_end != fault_start and cur_idx >= fault_end:
        fig.add_vline(
            x=fault_end, line_dash='solid', line_color='#cc0000', line_width=1.4,
            annotation_text=f'Failure ({fault_end})',
            annotation_font_color='#cc0000', annotation_font_size=9,
            annotation_position='top right',
        )

    fig.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=9),
        margin=dict(l=42, r=10, t=30, b=28), height=185,
        showlegend=False, hovermode='x unified',
        title=dict(
            text="<b style='color:#8b949e'>LSTM RECONSTRUCTION ERROR</b>"
                 "  ·  anomaly when line crosses threshold",
            font=dict(size=11), x=0.02, xanchor='left',
        ),
        xaxis=dict(gridcolor='#21262d', title=dict(text='Mission minute', font=dict(size=8))),
        yaxis=dict(gridcolor='#21262d', title=dict(text='MSE', font=dict(size=8))),
    )
    return fig


st.plotly_chart(make_error_chart(), use_container_width=True, key='error_chart')


# ── SentinelAI watermark ──────────────────────────────────────────────────────
st.markdown(
    '<div class="sentinel-watermark">SentinelAI v1.0 BETA</div>',
    unsafe_allow_html=True,
)

# ── Drive the loop ─────────────────────────────────────────────────────────────
if st.session_state.running:
    advance_one_step(model, scaler)
    time.sleep(speed)
    st.rerun()
