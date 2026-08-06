"""
Feature engineering pipeline.
Every feature is derivable from the six inputs available in `december-chart-inputs.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "posted_rate"
RATE_PER_MILE = "rate_per_mile"

DISTANCE_LABELS = ["very_short", "short", "medium", "long", "very_long", "extreme"]

# Where these numbers come from -> check the notebook, section "Distance
# distribution & binning": it plots the distribution and derives these edges.
DEFAULT_DISTANCE_BINS = [0.0, 431.0, 688.0, 953.0, 1323.0, 1968.0, np.inf]

FEATURE_GROUPS: dict[str, list[str]] = {
    "shipment": ["distance", "weight", "equipment", "distance_band", "weight_per_mile"],
    "geography": ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                  "lat_delta", "lon_delta", "heading_north", "heading_east"],
    "weekly": ["dow_sin", "dow_cos", "is_weekend"],
    "seasonal": ["doy_sin", "doy_cos"],
}
CATEGORICAL = ["equipment", "distance_band"]


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a challenge CSV, parsing `date` as a real datetime."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair data-quality defects found during exploration.
    """
    result = frame.copy()
    result["weight"] = result["weight"].abs()
    result["weight"] = result["weight"].fillna(result["weight"].median())
    return result


def build_city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Map every city name to its coordinates (each city has exactly one pair).
    """
    records: list[pd.DataFrame] = []
    for frame in frames:
        for role in ("pickup", "delivery"):
            columns = [role, f"{role}_lat", f"{role}_lon"]
            if not set(columns) <= set(frame.columns):
                continue
            records.append(frame[columns].rename(
                columns={role: "city", f"{role}_lat": "lat", f"{role}_lon": "lon"}))
    if not records:
        raise ValueError("no frame carried both city names and coordinates")
    return pd.concat(records).dropna().drop_duplicates("city").set_index("city")


def attach_coordinates(frame: pd.DataFrame, gazetteer: pd.DataFrame) -> pd.DataFrame:
    """Fill missing pickup/delivery coordinates from the city lookup."""
    result = frame.copy()
    for role in ("pickup", "delivery"):
        if f"{role}_lat" in result.columns:
            continue
        result[f"{role}_lat"] = result[role].map(gazetteer["lat"])
        result[f"{role}_lon"] = result[role].map(gazetteer["lon"])
    unknown = result[["pickup_lat", "delivery_lat"]].isna().any(axis=1).sum()
    if unknown:
        raise KeyError(f"{unknown} row(s) reference a city absent from the gazetteer")
    return result


def _cyclical(values: pd.Series, period: float, name: str) -> pd.DataFrame:
    """Project a periodic value onto a circle so the cycle wraps around.

    Encoded as an integer, December (12) and January (1) sit at opposite ends of
    the scale even though they are adjacent in time. The sine/cosine pair
    restores that adjacency and keeps every value inside [-1, 1].
    """
    angle = 2.0 * np.pi * values / period
    return pd.DataFrame({f"{name}_sin": np.sin(angle), f"{name}_cos": np.cos(angle)},
                        index=values.index)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived from `date` alone."""
    result = frame.copy()
    date = result["date"]
    result = pd.concat([result, _cyclical(date.dt.dayofweek, 7.0, "dow")], axis=1)
    result = pd.concat([result, _cyclical(date.dt.dayofyear, 365.25, "doy")], axis=1)
    result["is_weekend"] = (date.dt.dayofweek >= 5).astype(int)
    return result


def add_lane_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Geographic features."""
    result = frame.copy()
    result["lat_delta"] = result["delivery_lat"] - result["pickup_lat"]
    result["lon_delta"] = result["delivery_lon"] - result["pickup_lon"]
    result["heading_north"] = (result["lat_delta"] > 0).astype(int)
    result["heading_east"] = (result["lon_delta"] > 0).astype(int)
    return result


def distance_bin_edges(frame: pd.DataFrame, n_bins: int = 6) -> list[float]:
    """Equal-frequency (quantile) bin edges derived from the training distance.
    """
    quantiles = frame["distance"].quantile(np.linspace(0, 1, n_bins + 1)).to_numpy()
    edges = [0.0, *quantiles[1:-1].round().tolist(), np.inf]
    return edges


def add_shipment_features(
    frame: pd.DataFrame, bin_edges: list[float] | None = None
) -> pd.DataFrame:
    """Shipment-level features, including the binned distance.
    """
    result = frame.copy()
    result["equipment"] = result["equipment"].astype("category")
    edges = bin_edges if bin_edges is not None else DEFAULT_DISTANCE_BINS
    result["distance_band"] = pd.cut(
        result["distance"], bins=edges, labels=DISTANCE_LABELS
    ).astype("category")
    result["weight_per_mile"] = result["weight"] / result["distance"]
    return result


def build_features(
    frame: pd.DataFrame,
    gazetteer: pd.DataFrame | None = None,
    bin_edges: list[float] | None = None,
) -> pd.DataFrame:
    """Run the full feature pipeline.
    """
    result = clean(frame)
    if gazetteer is not None:
        result = attach_coordinates(result, gazetteer)
    result = add_calendar_features(result)
    result = add_lane_features(result)
    result = add_shipment_features(result, bin_edges=bin_edges)
    return result


def add_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the normalised target."""
    if TARGET not in frame.columns:
        raise KeyError(f"{TARGET} missing: this frame carries no labels")
    result = frame.copy()
    result[RATE_PER_MILE] = result[TARGET] / result["distance"]
    return result


def feature_names(*groups: str) -> list[str]:
    """Collect the feature columns for the named groups (all groups by default)."""
    selected = groups or tuple(FEATURE_GROUPS)
    unknown = set(selected) - set(FEATURE_GROUPS)
    if unknown:
        raise KeyError(f"unknown feature group(s): {sorted(unknown)}")
    return [column for group in selected for column in FEATURE_GROUPS[group]]
