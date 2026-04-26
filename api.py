from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import joblib
from pathlib import Path
from tensorflow import keras

# Load models and scaler
MODEL_DIR = Path(__file__).parent / "models"
ISOLATION_FOREST_MODEL = MODEL_DIR / "isolation_forest.joblib"
SCALER_MODEL = MODEL_DIR / "scaler.joblib"
AUTOENCODER_MODEL_DIR = MODEL_DIR / "autoencoder" / "autoencoder.keras"

isolation_forest = joblib.load(ISOLATION_FOREST_MODEL)
scaler = joblib.load(SCALER_MODEL)
autoencoder = keras.models.load_model(AUTOENCODER_MODEL_DIR)

# Load threshold from metrics (assuming it's saved)
import json
METRICS_PATH = Path(__file__).parent / "logs" / "metrics.json"
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)
autoencoder_threshold = metrics["autoencoder_threshold"]

app = FastAPI(title="Satellite Telemetry Anomaly Detection API", version="1.0.0")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

class TelemetryInput(BaseModel):
    BatteryTemperature: float
    BusVoltage: float
    TotalBusCurrent: float
    WheelRPM: float
    WheelTemperature: float

@app.post("/predict")
def predict_anomaly(telemetry: TelemetryInput) -> Dict[str, Any]:
    try:
        # Convert input to array
        data = np.array([[telemetry.BatteryTemperature, telemetry.BusVoltage,
                         telemetry.TotalBusCurrent, telemetry.WheelRPM, telemetry.WheelTemperature]])
        
        # Scale the data
        scaled_data = scaler.transform(data)
        
        # Isolation Forest prediction
        if_prediction = isolation_forest.predict(scaled_data)[0]
        if_score = isolation_forest.decision_function(scaled_data)[0]
        if_anomaly = if_prediction == -1
        
        # Autoencoder prediction
        reconstructed = autoencoder.predict(scaled_data)
        ae_error = np.mean(np.square(scaled_data - reconstructed), axis=1)[0]
        ae_anomaly = ae_error > autoencoder_threshold
        
        return {
            "isolation_forest": {
                "anomaly_score": float(if_score),
                "is_anomaly": bool(if_anomaly)
            },
            "autoencoder": {
                "anomaly_score": float(ae_error),
                "is_anomaly": bool(ae_anomaly)
            },
            "overall_anomaly": bool(if_anomaly or ae_anomaly)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/health")
def health():
    return {"status": "healthy"}
