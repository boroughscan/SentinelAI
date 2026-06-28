# Spacecraft Health Monitor

Predictive anomaly detection dashboard for simulated Orion spacecraft telemetry.

## Stack
Python · Streamlit · Plotly · Scikit-learn · Keras

## Project structure
```
spacecraft-monitor/
├── data/           sensor data, generator, datasets
├── models/         trained ML models
├── dashboard/      Streamlit app
├── docs/           design notes
├── notebooks/      exploratory analysis
└── README.md
```

## Week 1 Day 1 — run the generator

```bash
cd spacecraft-monitor
python data/generate_datasets.py
```

Generates `data/normal_telemetry.csv`, `data/fault_telemetry.csv`, and
`day1_telemetry_comparison.png`.
