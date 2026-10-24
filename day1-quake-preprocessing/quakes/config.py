from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"


# Numeric columns used by the ML preprocessor
NUMERIC: list[str] = [
    "lon",
    "lat",
    "depth_km",
    "nst",
    "gap",
    "dmin",
    "rms",
    "hour",
    "dayofweek",
    "update_lag_hours",
    "is_reviewed",
    "nst_missing",
    "abs_lat",
    "is_shallow",
]


# Categorical columns
NOMINAL: list[str] = [
    "region",
    "magType",
]


# Columns that may cause leakage
LEAKY: list[str] = [
    "title",
    "sig",
    "mmi",
    "cdi",
    "felt",
    "alert",
]