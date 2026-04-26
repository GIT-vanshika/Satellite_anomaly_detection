import numpy as np
from tensorflow import keras


def estimate_false_positive_rate(predictions: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    if len(predictions) == 0:
        return 0.0
    false_positives = np.sum(predictions == -1)
    return float(false_positives / len(predictions))


def compute_reconstruction_errors(model: keras.Model, X: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(X)
    return np.mean(np.square(X - reconstructed), axis=1)


def detect_autoencoder_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(errors > threshold, -1, 1)
