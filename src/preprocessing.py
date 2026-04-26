import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.interpolate(method="time", limit_direction="both")
    cleaned = cleaned.ffill().bfill()
    return cleaned


def standardize_features(df: pd.DataFrame, scaler: StandardScaler = None) -> tuple[pd.DataFrame, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    return scaled_df, scaler


def split_train_validation(df: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(df) * (1 - validation_fraction))
    train_df = df.iloc[:split].copy()
    validation_df = df.iloc[split:].copy()
    return train_df, validation_df
