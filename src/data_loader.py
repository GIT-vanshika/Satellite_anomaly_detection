import pandas as pd
from pathlib import Path

from .config import SENSOR_FILES


def load_csv_sensor(sensor_name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["timestamp", "value"], parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.rename(columns={"value": sensor_name})
    return df


def merge_telemetry_data() -> pd.DataFrame:
    sensor_frames = [load_csv_sensor(sensor_name, path) for sensor_name, path in SENSOR_FILES.items()]
    merged = pd.concat(sensor_frames, axis=1, join="outer")
    merged.index.name = "timestamp"
    return merged
