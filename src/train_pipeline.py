from pathlib import Path

import numpy as np

from .config import (
    AUTOENCODER_MODEL_DIR,
    ISOLATION_FOREST_MODEL,
    LOG_PATH,
    LOG_DIR,
    METRICS_PATH,
    MODEL_DIR,
    SCALER_MODEL,
)
from .data_loader import merge_telemetry_data
from .evaluate import (
    compute_reconstruction_errors,
    detect_autoencoder_anomalies,
    estimate_false_positive_rate,
)
from .models import (
    build_autoencoder,
    save_isolation_forest,
    save_scaler,
    train_autoencoder,
    train_isolation_forest,
)
from .preprocessing import clean_missing_values, split_train_validation, standardize_features
from .utils import configure_logger, ensure_dir, save_metrics


def main() -> None:
    ensure_dir(MODEL_DIR)
    ensure_dir(LOG_DIR)
    logger = configure_logger(LOG_PATH)

    logger.info("Loading multivariate telemetry data...")
    merged_df = merge_telemetry_data()
    logger.info("Loaded %d timestamp rows and %d feature columns.", merged_df.shape[0], merged_df.shape[1])

    logger.info("Cleaning missing values...")
    cleaned_df = clean_missing_values(merged_df)
    missing_after = int(cleaned_df.isna().sum().sum())
    logger.info("Missing values remaining after cleanup: %d", missing_after)

    logger.info("Sampling dataset to 300,000 rows for efficiency...")
    cleaned_df = cleaned_df.sample(300000, random_state=42)
    logger.info("Sampled dataset shape: %s", cleaned_df.shape)

    logger.info("Standardizing features...")
    standardized_df, scaler = standardize_features(cleaned_df)

    logger.info("Splitting data into training and validation sets...")
    train_df, validation_df = split_train_validation(standardized_df, validation_fraction=0.2)
    logger.info(
        "Training rows: %d, Validation rows: %d",
        train_df.shape[0],
        validation_df.shape[0],
    )

    logger.info("Training Isolation Forest model...")
    isolation_model = train_isolation_forest(train_df.values, contamination=0.01)

    logger.info("Building autoencoder model...")
    autoencoder = build_autoencoder(input_dim=train_df.shape[1])
    autoencoder, threshold, history = train_autoencoder(
        autoencoder,
        train_df.values,
        validation_df.values,
        epochs=20,
        batch_size=128,
    )
    logger.info("Autoencoder training completed. Threshold=%0.6f", threshold)

    logger.info("Evaluating false positive rate on validation data...")
    validation_predictions = isolation_model.predict(validation_df.values)
    isolation_fpr = estimate_false_positive_rate(validation_predictions)

    reconstruction_errors = compute_reconstruction_errors(autoencoder, validation_df.values)
    autoencoder_predictions = detect_autoencoder_anomalies(reconstruction_errors, threshold)
    autoencoder_fpr = estimate_false_positive_rate(autoencoder_predictions)

    logger.info("Isolation Forest estimated false positive rate: %.4f", isolation_fpr)
    logger.info("Autoencoder estimated false positive rate: %.4f", autoencoder_fpr)

    logger.info("Saving models and scaler...")
    save_isolation_forest(isolation_model, str(ISOLATION_FOREST_MODEL))
    save_scaler(scaler, str(SCALER_MODEL))
    AUTOENCODER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    autoencoder.save(AUTOENCODER_MODEL_DIR / "autoencoder.keras")

    metrics = {
        "rows_total": int(standardized_df.shape[0]),
        "features": int(standardized_df.shape[1]),
        "train_rows": int(train_df.shape[0]),
        "validation_rows": int(validation_df.shape[0]),
        "isolation_forest_false_positive_rate": float(isolation_fpr),
        "autoencoder_false_positive_rate": float(autoencoder_fpr),
        "autoencoder_threshold": float(threshold),
    }
    save_metrics(metrics, METRICS_PATH)
    logger.info("Pipeline completed successfully. Metrics saved to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
