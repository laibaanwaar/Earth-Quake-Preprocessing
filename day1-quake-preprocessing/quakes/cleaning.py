"""Task 3: cleaning."""

import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'time' and 'updated' (epoch milliseconds) to UTC datetimes."""

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True
    )

    df["updated"] = pd.to_datetime(
        df["updated"],
        unit="ms",
        utc=True
    )

    return df


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One row per 'id', keeping the row with the greatest 'updated'."""

    df = df.copy()

    df = df.sort_values("updated")

    df = df.drop_duplicates(
        subset="id",
        keep="last"
    )

    return df


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise 'type' (strip, lowercase) and keep only 'earthquake'."""

    df = df.copy()

    df["type"] = (
        df["type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df = df[df["type"] == "earthquake"]

    return df


def extract_region(place: pd.Series) -> pd.Series:
    """Text after the last comma, or the whole string if there is no comma.
    Missing places become 'Unknown'."""

    place = place.fillna("Unknown").astype("string")

    region = (
        place
        .str.split(",")
        .str[-1]
        .str.strip()
    )

    return region


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies outside [Q1 - k*IQR, Q3 + k*IQR]."""

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    mask = (s < lower_bound) | (s > upper_bound)

    return mask


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 'mag' is missing."""

    df = df.copy()

    df = df.dropna(subset=["mag"])

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Chain the steps above (think about the order) and add a 'region' column."""

    df = epoch_ms_to_datetime(df)

    df = dedupe_latest(df)

    df = keep_earthquakes(df)

    df = drop_missing_target(df)

    df = df.copy()

    df["region"] = extract_region(df["place"])

    return df