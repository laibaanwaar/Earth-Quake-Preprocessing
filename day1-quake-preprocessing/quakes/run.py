"""Task 5: run the whole pipeline end to end.

Steps (see LAB_GUIDE.md):
 1. fetch all_week (fall back to config.FALLBACK_PATH if the network fails)
 2. merge any data/stream/*.jsonl rows, then dedupe_latest
 3. clean, engineer features, drop leaky columns
 4. split, fit_transform on train only, transform test
 5. print the report, save train.csv, test.csv, preprocessor.joblib
"""

import json

import joblib
import pandas as pd

from . import config
from .fetch import fetch_feed, geojson_to_df
from .cleaning import clean, dedupe_latest
from .features import (
    add_time_features,
    add_quality_features,
    add_location_features,
    group_rare,
    add_target,
    drop_leaky_columns,
)
from .transform import split_data, build_preprocessor


def main() -> None:

    # -------------------------------------------------
    # 1. Fetch live all_week USGS data
    # -------------------------------------------------
    try:
        print("Fetching live USGS all_week feed...")

        payload = fetch_feed("all_week")
        raw_df = geojson_to_df(payload)

        print("Live feed loaded successfully.")

    except Exception as e:
        print(f"Live feed failed: {e}")
        print("Using fallback file...")

        with open(config.FALLBACK_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_df = geojson_to_df(payload)

    raw_rows = len(raw_df)

    # -------------------------------------------------
    # 2. Read and merge stream JSONL files
    # -------------------------------------------------
    stream_frames = []

    for path in config.STREAM_DIR.glob("*.jsonl"):
        try:
            stream_df = pd.read_json(
                path,
                orient="records",
                lines=True
            )

            stream_frames.append(stream_df)

        except Exception as e:
            print(f"Could not read {path.name}: {e}")

    if stream_frames:
        stream_df = pd.concat(
            stream_frames,
            ignore_index=True
        )

        stream_rows = len(stream_df)

        combined_df = pd.concat(
            [raw_df, stream_df],
            ignore_index=True
        )

    else:
        stream_rows = 0
        combined_df = raw_df.copy()

    # -------------------------------------------------
    # Count events whose updated value changed
    # -------------------------------------------------
    changed_events = 0

    if (
        "id" in combined_df.columns
        and "updated" in combined_df.columns
    ):
        changed_events = (
            combined_df
            .groupby("id")["updated"]
            .nunique()
            .gt(1)
            .sum()
        )

    # Keep latest version of each event
    combined_df = dedupe_latest(combined_df)

    # -------------------------------------------------
    # 3. Cleaning
    # -------------------------------------------------
    cleaned_df = clean(combined_df)

    rows_after_cleaning = len(cleaned_df)

    # -------------------------------------------------
    # Feature engineering
    # -------------------------------------------------
    df = add_time_features(cleaned_df)

    df = add_quality_features(df)

    df = add_location_features(df)

    # Reduce region cardinality
    df["region"] = group_rare(
        df["region"],
        top_k=15
    )

    # Create classification target
    df = add_target(df)

    # Remove leakage columns
    df = drop_leaky_columns(df)

    # -------------------------------------------------
    # 4. Train/test split
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(df)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(
        scaler="robust"
    )

    # IMPORTANT:
    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Only transform the test data
    X_test_processed = preprocessor.transform(
        X_test
    )

    # -------------------------------------------------
    # Get transformed feature names
    # -------------------------------------------------
    feature_names = preprocessor.get_feature_names_out()

    train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names
    )

    # Add target back to saved datasets
    train_processed[config.TARGET] = (
        y_train.reset_index(drop=True)
    )

    test_processed[config.TARGET] = (
        y_test.reset_index(drop=True)
    )

    # -------------------------------------------------
    # 5. Print report
    # -------------------------------------------------
    print("\n========== PIPELINE REPORT ==========")

    print(f"Raw rows: {raw_rows}")

    print(
        f"Rows after cleaning: "
        f"{rows_after_cleaning}"
    )

    print(
        f"Stream rows merged: "
        f"{stream_rows}"
    )

    print(
        f"Events whose updated changed: "
        f"{changed_events}"
    )

    print(
        f"Train shape: "
        f"{train_processed.shape}"
    )

    print(
        f"Test shape: "
        f"{test_processed.shape}"
    )

    print(
        f"Train positive-class rate: "
        f"{y_train.mean():.3f}"
    )

    print(
        f"Test positive-class rate: "
        f"{y_test.mean():.3f}"
    )

    print("=====================================\n")

    # -------------------------------------------------
    # 6. Save outputs
    # -------------------------------------------------
    config.PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = (
        config.PROCESSED_DIR / "train.csv"
    )

    test_path = (
        config.PROCESSED_DIR / "test.csv"
    )

    preprocessor_path = (
        config.PROCESSED_DIR /
        "preprocessor.joblib"
    )

    train_processed.to_csv(
        train_path,
        index=False
    )

    test_processed.to_csv(
        test_path,
        index=False
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    print("Saved:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {preprocessor_path}")


if __name__ == "__main__":
    main()