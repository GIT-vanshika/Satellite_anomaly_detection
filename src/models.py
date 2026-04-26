import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


def train_isolation_forest(X: np.ndarray, contamination: float = 0.01, random_state: int = 42) -> IsolationForest:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    return model


def build_autoencoder(input_dim: int, encoding_dim: int = 16) -> keras.Model:
    encoder = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(encoding_dim, activation="relu"),
        ],
        name="encoder",
    )
    decoder = keras.Sequential(
        [
            layers.Input(shape=(encoding_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="linear"),
        ],
        name="decoder",
    )
    autoencoder = keras.Sequential([encoder, decoder], name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(
    autoencoder: keras.Model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 128,
) -> tuple[keras.Model, float, keras.callbacks.History]:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    history = autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )
    reconstructed = autoencoder.predict(X_val)
    mse = np.mean(np.square(X_val - reconstructed), axis=1)
    threshold = float(mse.mean() + 3 * mse.std())
    return autoencoder, threshold, history


def save_isolation_forest(model: BaseEstimator, path: str) -> None:
    import joblib

    joblib.dump(model, path)


def save_scaler(scaler: StandardScaler, path: str) -> None:
    import joblib

    joblib.dump(scaler, path)
