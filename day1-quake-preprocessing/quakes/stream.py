"""Task 2: simulate streaming by polling a feed on a timer."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from quakes.fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return only rows whose (id, updated) pair is not in `seen`.

    Add the new pairs to `seen` (modify the set in place).
    """

    # Create (id, updated) pairs for each row
    pairs = list(zip(df["id"], df["updated"]))

    # Keep only rows whose pair is not already in seen
    mask = [pair not in seen for pair in pairs]

    new_df = df.loc[mask].copy()

    # Add new pairs into seen
    for pair in zip(new_df["id"], new_df["updated"]):
        seen.add(pair)

    return new_df


def run_stream(feed: str, interval_s: int, duration_s: int, out_path: str) -> None:
    """Poll `feed` every `interval_s` seconds for `duration_s` seconds.

    Each poll: fetch, parse, keep unseen rows, append them to `out_path`
    as JSON lines, and print e.g. "[14:02:11] 3 new / 12 fetched".
    A failed poll must not stop the loop.
    """

    seen = set()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    while time.time() - start_time < duration_s:

        try:
            # Fetch GeoJSON data
            payload = fetch_feed(feed)

            # Convert GeoJSON into DataFrame
            df = geojson_to_df(payload)

            # Keep only unseen events
            new_df = filter_unseen(df, seen)

            # Append new rows to JSONL file
            if not new_df.empty:
                with open(out_file, "a", encoding="utf-8") as f:
                    new_df.to_json(
                        f,
                        orient="records",
                        lines=True
                    )
                    f.write("\n")

            # Print status
            current_time = datetime.now().strftime("%H:%M:%S")

            print(
                f"[{current_time}] "
                f"{len(new_df)} new / {len(df)} fetched"
            )

        except Exception as e:
            # One failed request should not stop the stream
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] Poll failed: {e}")

        # Wait before next poll
        time.sleep(interval_s)


def main() -> None:
    """argparse: --feed (default all_hour), --interval (60), --duration (2400),
    --out (data/stream/stream.jsonl), then call run_stream."""

    parser = argparse.ArgumentParser(
        description="Poll the USGS earthquake feed and save new events."
    )

    parser.add_argument(
        "--feed",
        default="all_hour"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=2400
    )

    parser.add_argument(
        "--out",
        default="data/stream/stream.jsonl"
    )

    args = parser.parse_args()

    run_stream(
        feed=args.feed,
        interval_s=args.interval,
        duration_s=args.duration,
        out_path=args.out
    )


if __name__ == "__main__":
    main()