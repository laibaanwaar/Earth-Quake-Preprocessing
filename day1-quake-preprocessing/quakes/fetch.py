"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""

import pandas as pd
import requests

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """GET {BASE}/{feed}.geojson and return the parsed dict.

    Use a timeout and raise_for_status(). A misspelled feed name returns
    HTTP 200 with a plain-text body, so raise a clear error if the body
    is not valid JSON.
    """

    # Build the complete USGS feed URL
    url = f"{BASE}/{feed}.geojson"

    # Send GET request to USGS
    response = requests.get(url, timeout=15)

    # Raise an error for HTTP errors such as 404 or 500
    response.raise_for_status()

    # Convert the response JSON into a Python dictionary
    try:
        payload = response.json()
    except ValueError as e:
        raise ValueError(
            f"Feed '{feed}' did not return valid JSON."
        ) from e

    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """One row per event.

    Columns: 'id' (top level of each feature), every key in 'properties',
    plus 'lon', 'lat', 'depth_km' from geometry.coordinates = [lon, lat, depth].
    """

    rows = []

    # Each item inside "features" represents one earthquake event
    for feature in payload.get("features", []):

        # Start a row with the top-level event ID
        row = {
            "id": feature.get("id")
        }

        # Add all values from the properties dictionary
        properties = feature.get("properties", {})
        row.update(properties)

        # Extract coordinates from geometry
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []

        # USGS coordinate order:
        # [longitude, latitude, depth_km]
        row["lon"] = coordinates[0] if len(coordinates) > 0 else None
        row["lat"] = coordinates[1] if len(coordinates) > 1 else None
        row["depth_km"] = coordinates[2] if len(coordinates) > 2 else None

        # Add the completed earthquake row to the list
        rows.append(row)

    # Convert list of dictionaries into a pandas DataFrame
    return pd.DataFrame(rows)