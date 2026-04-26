from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from pathlib import Path
from tensorflow import keras
import json

app = Flask(__name__)

# Load models and scaler
MODEL_DIR = Path(__file__).parent / "models"
ISOLATION_FOREST_MODEL = MODEL_DIR / "isolation_forest.joblib"
SCALER_MODEL = MODEL_DIR / "scaler.joblib"
AUTOENCODER_MODEL_DIR = MODEL_DIR / "autoencoder" / "autoencoder.keras"

isolation_forest = joblib.load(ISOLATION_FOREST_MODEL)
scaler = joblib.load(SCALER_MODEL)
autoencoder = keras.models.load_model(AUTOENCODER_MODEL_DIR)

METRICS_PATH = Path(__file__).parent / "logs" / "metrics.json"
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)
autoencoder_threshold = metrics["autoencoder_threshold"]

REQUIRED_FIELDS = [
    "BatteryTemperature",
    "BusVoltage",
    "TotalBusCurrent",
    "WheelRPM",
    "WheelTemperature",
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict_anomaly():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "JSON payload is required."}), 400

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        return jsonify({"error": "Missing telemetry fields.", "missing": missing}), 400

    try:
        data = np.array([
            [
                float(payload["BatteryTemperature"]),
                float(payload["BusVoltage"]),
                float(payload["TotalBusCurrent"]),
                float(payload["WheelRPM"]),
                float(payload["WheelTemperature"]),
            ]
        ])

        scaled_data = scaler.transform(data)
        if_prediction = isolation_forest.predict(scaled_data)[0]
        if_score = isolation_forest.decision_function(scaled_data)[0]
        if_anomaly = if_prediction == -1

        reconstructed = autoencoder.predict(scaled_data)
        ae_error = np.mean(np.square(scaled_data - reconstructed), axis=1)[0]
        ae_anomaly = ae_error > autoencoder_threshold

        return jsonify(
            {
                "isolation_forest": {
                    "anomaly_score": float(if_score),
                    "is_anomaly": bool(if_anomaly),
                },
                "autoencoder": {
                    "anomaly_score": float(ae_error),
                    "is_anomaly": bool(ae_anomaly),
                },
                "overall_anomaly": bool(if_anomaly or ae_anomaly),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=True)
