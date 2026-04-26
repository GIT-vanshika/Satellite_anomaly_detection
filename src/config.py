from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

SENSOR_FILES = {
    "BatteryTemperature": DATA_DIR / "BatteryTemperature.csv",
    "BusVoltage": DATA_DIR / "BusVoltage.csv",
    "TotalBusCurrent": DATA_DIR / "TotalBusCurrent.csv",
    "WheelRPM": DATA_DIR / "WheelRPM.csv",
    "WheelTemperature": DATA_DIR / "WheelTemperature.csv",
}

ISOLATION_FOREST_MODEL = MODEL_DIR / "isolation_forest.joblib"
SCALER_MODEL = MODEL_DIR / "scaler.joblib"
AUTOENCODER_MODEL_DIR = MODEL_DIR / "autoencoder"
METRICS_PATH = LOG_DIR / "metrics.json"
LOG_PATH = LOG_DIR / "pipeline.log"
