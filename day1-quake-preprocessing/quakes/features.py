"""Task 4: feature engineering."""

import pandas as pd

from quakes import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'hour' and 'dayofweek' (UTC) from 'time'."""

    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    return df


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add:
    update_lag_hours = hours between 'time' and 'updated'
    is_reviewed      = 1 if status == 'reviewed' else 0
    nst_missing      = 1 if 'nst' is missing else 0
    """

    df = df.copy()

    # Difference between update time and original event time
    df["update_lag_hours"] = (
        (df["updated"] - df["time"]).dt.total_seconds() / 3600
    )

    # 1 if earthquake status is reviewed, otherwise 0
    df["is_reviewed"] = (
        df["status"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("reviewed")
        .astype(int)
    )

    # 1 if nst is missing, otherwise 0
    df["nst_missing"] = df["nst"].isna().astype(int)

    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'abs_lat' and 'is_shallow' (1 if depth_km < 70 else 0)."""

    df = df.copy()

    # Absolute latitude
    df["abs_lat"] = df["lat"].abs()

    # Earthquake is shallow if depth is less than 70 km
    df["is_shallow"] = (df["depth_km"] < 70).astype(int)

    return df


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top_k most frequent values; replace all others with 'Other'."""

    top_values = s.value_counts().nlargest(top_k).index

    return s.where(s.isin(top_values), "Other")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'big_quake' = 1 if mag >= 4.5 else 0."""

    df = df.copy()

    df["big_quake"] = (df["mag"] >= 4.5).astype(int)

    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every column listed in config.LEAKY (ignore ones that are absent)."""

    return df.drop(
        columns=config.LEAKY,
        errors="ignore"
    )