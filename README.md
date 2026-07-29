# Satellite Telemetry Anomaly Detection

End-to-end machine learning project for detecting anomalous satellite telemetry patterns from multivariate sensor readings. The project includes data ingestion, preprocessing, unsupervised model training, evaluation, saved model artifacts, and REST API deployment through FastAPI, Flask, and Docker.

## Why This Project Matters

Satellites generate continuous telemetry from onboard subsystems such as batteries, bus voltage, current draw, and reaction wheels. Unexpected shifts in these signals can indicate component degradation, abnormal operating conditions, or early-stage failures. This project builds an unsupervised anomaly detection pipeline that can flag unusual telemetry without requiring manually labeled fault examples.

## Key Highlights

- Built a modular Python training pipeline for satellite telemetry anomaly detection.
- Combined two unsupervised approaches: Isolation Forest and a TensorFlow/Keras autoencoder.
- Processed and standardized five telemetry channels into a unified timestamp-indexed dataset.
- Trained on a 300,000-row sampled dataset with an 80/20 train-validation split.
- Saved production-ready artifacts: scaler, Isolation Forest model, autoencoder model, logs, and metrics.
- Exposed predictions through FastAPI and Flask APIs.
- Added Docker and Docker Compose support for containerized API deployment.

## Dataset

The telemetry data is sourced from:

[Satellite Telemetry Anomaly Detection - Data Source](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection/tree/master/Data)

This repository uses the following sensor files from the `Data/` directory:

| Feature | File |
| --- | --- |
| Battery temperature | `BatteryTemperature.csv` |
| Bus voltage | `BusVoltage.csv` |
| Total bus current | `TotalBusCurrent.csv` |
| Wheel RPM | `WheelRPM.csv` |
| Wheel temperature | `WheelTemperature.csv` |

Each sensor file is loaded as a timestamp-value series, sorted by timestamp, and merged into one multivariate telemetry dataframe.

## Tech Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, TensorFlow/Keras
- **Model persistence:** joblib, Keras model format
- **API:** FastAPI, Flask
- **Serving:** Uvicorn
- **Templating:** Jinja2
- **Deployment:** Docker, Docker Compose

## Project Structure

```text
.
|-- Data/                    # Source telemetry CSV files
|-- NoteBook/                # Exploratory notebook
|-- logs/
|   |-- metrics.json         # Saved training/evaluation metrics
|   `-- pipeline.log         # Pipeline execution log
|-- models/
|   |-- autoencoder/         # Saved Keras autoencoder
|   |-- isolation_forest.joblib
|   `-- scaler.joblib
|-- src/
|   |-- config.py            # Project paths and feature configuration
|   |-- data_loader.py       # Sensor CSV loading and merging
|   |-- preprocessing.py     # Missing value cleanup, scaling, split
|   |-- models.py            # Isolation Forest and autoencoder training
|   |-- evaluate.py          # Reconstruction error and FPR utilities
|   |-- train_pipeline.py    # End-to-end training pipeline
|   `-- utils.py             # Logging and metrics helpers
|-- templates/
|   `-- index.html           # Web UI template
|-- api.py                   # FastAPI prediction service
|-- flask_api.py             # Flask prediction service
|-- run_pipeline.py          # Pipeline entry point
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

## Methodology

1. **Data ingestion**
   - Load each sensor CSV with timestamp parsing.
   - Sort each stream by timestamp.
   - Merge all telemetry streams into one timestamp-indexed dataframe.

2. **Preprocessing**
   - Fill missing values using time-based interpolation.
   - Apply forward-fill and backward-fill cleanup for remaining gaps.
   - Standardize features using `StandardScaler`.
   - Split data into training and validation sets.

3. **Modeling**
   - Train an `IsolationForest` with `contamination=0.01`.
   - Train a dense autoencoder using reconstruction loss.
   - Set the autoencoder anomaly threshold as:

   ```text
   validation_mse_mean + 3 * validation_mse_std
   ```

4. **Inference**
   - Scale incoming telemetry with the saved scaler.
   - Generate anomaly decisions from both models.
   - Return model-level scores and an overall anomaly flag.

## Training Metrics

The latest saved run in `logs/metrics.json` reports:

| Metric | Value |
| --- | ---: |
| Sampled rows used | 300,000 |
| Feature count | 5 |
| Training rows | 240,000 |
| Validation rows | 60,000 |
| Isolation Forest estimated false positive rate | 0.0094 |
| Autoencoder estimated false positive rate | 0.0047 |
| Autoencoder threshold | 0.001207 |

Note: The dataset is treated as unlabeled telemetry, so the reported false positive rates are estimated from model anomaly predictions on validation data rather than from ground-truth fault labels.

## API Input Features

The prediction APIs expect these five numeric fields:

```json
{
  "BatteryTemperature": 3.8,
  "BusVoltage": 32.0,
  "TotalBusCurrent": 1266.0,
  "WheelRPM": 1000.0,
  "WheelTemperature": 25.0
}
```

## Installation

Clone the repository:

```bash
git clone https://github.com/GIT-vanshika/Satellite_anomaly_detection.git
cd Satellite_anomaly_detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Training Pipeline

```bash
python run_pipeline.py
```

Expected generated or updated artifacts:

- `models/isolation_forest.joblib`
- `models/scaler.joblib`
- `models/autoencoder/autoencoder.keras`
- `logs/pipeline.log`
- `logs/metrics.json`

## Run The FastAPI Service

```bash
uvicorn api:app --reload
```

Open the API home page:

```text
http://127.0.0.1:8000/
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "BatteryTemperature": 3.8,
    "BusVoltage": 32.0,
    "TotalBusCurrent": 1266.0,
    "WheelRPM": 1000.0,
    "WheelTemperature": 25.0
  }'
```

Example response:

```json
{
  "isolation_forest": {
    "anomaly_score": 0.1453,
    "is_anomaly": false
  },
  "autoencoder": {
    "anomaly_score": 0.0008,
    "is_anomaly": false
  },
  "overall_anomaly": false
}
```

## Run The Flask Service

```bash
python flask_api.py
```

By default, the Flask app runs at:

```text
http://127.0.0.1:8001/
```

## Run With Docker

Build and start the FastAPI service:

```bash
docker compose up --build
```

The container exposes the API at:

```text
http://127.0.0.1:8000/
```

Stop the service:

```bash
docker compose down
```

## What Recruiters Can Notice

- **Machine learning depth:** uses both tree-based anomaly detection and neural reconstruction-based anomaly detection.
- **Production readiness:** includes reusable pipeline modules, saved model artifacts, API serving, health checks, and Docker deployment.
- **Clean software structure:** separates configuration, loading, preprocessing, modeling, evaluation, and serving.
- **Practical evaluation:** tracks validation-set anomaly rates and saves reproducible metrics.
- **Real-world framing:** solves an operations-style telemetry monitoring problem where labeled failures may be rare or unavailable.

## Limitations And Future Improvements

- Add labeled anomaly examples, if available, for precision, recall, F1-score, and confusion matrix evaluation.
- Add automated tests for preprocessing, model loading, and API responses.
- Add request validation ranges based on domain-specific sensor limits.
- Compare additional anomaly detection methods such as One-Class SVM and Local Outlier Factor.
- Add experiment tracking for model versions and hyperparameter runs.

## Author

Vanshika
