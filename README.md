
# SensorGuard

SensorGuard is a reproducible data and machine-learning pipeline that generates equipment telemetry, validates and cleans bad records, stores trusted data in SQLite, and detects unusual readings with IsolationForest.

## What it demonstrates

- Python data validation and feature engineering
- CSV and SQLite data pipelines
- SQL aggregation
- Baseline comparison using precision and recall
- Multivariate anomaly detection
- Reproducible visual reports

## Setup

Use Python 3.12.13 and install the pinned dependencies:

```bash
pip install pandas==3.0.6 scikit-learn==1.9.1 matplotlib==3.11.2
```

## Run

```bash
python sensorguard.py
```

## Outputs

- `raw_telemetry.csv`: generated telemetry before cleaning
- `clean_telemetry.csv`: validated data with predictions and scores
- `anomalies.csv`: records flagged by the ML detector
- `sensorguard.db`: SQLite database containing the cleaned telemetry table
- `raw_telemetry.png`: raw temperature and vibration plots
- `anomaly_report.png`: final anomaly visualization

## Evaluation

The project first tests a temperature-only threshold. That baseline misses the vibration-only and low-temperature events. IsolationForest then evaluates temperature, vibration, pressure, and temperature change together.

The data is synthetic and deterministic, so the pipeline can be reproduced without external credentials or dataset licensing.
