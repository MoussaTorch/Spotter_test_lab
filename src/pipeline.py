"""End-to-end run: build features, train, predict, write the submission files.

This is the single script to run. The notebook only analyses; this produces the
graded outputs. Run with `python -m src.pipeline`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .features import RATE_PER_MILE
from .model import make_model
from .registry import save_model

# Final features, fixed from the sweep. See README.
FINAL_FEATURES = [
    "distance", "weight", "equipment", "weight_per_mile",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]

# December output must keep exactly these seven columns, in this order (score.py).
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def run(data_dir: Path, output_dir: Path) -> None:
    """Train on the labelled data and write both prediction files."""
    train_raw = F.load_dataset(data_dir / "train-test.csv")
    validation_raw = F.load_dataset(data_dir / "validation.csv")
    december_raw = F.load_dataset(data_dir / "december-chart-inputs.csv")

    # Learn the two lookups on the training data, reuse everywhere.
    gazetteer = F.build_city_coordinates(train_raw, validation_raw)
    bin_edges = F.distance_bin_edges(train_raw)

    train = F.add_target(F.build_features(train_raw, bin_edges=bin_edges))
    validation = F.build_features(validation_raw, gazetteer=gazetteer, bin_edges=bin_edges)
    december = F.build_features(december_raw, gazetteer=gazetteer, bin_edges=bin_edges)

    model = make_model(FINAL_FEATURES)
    model.fit(train[FINAL_FEATURES], np.log(train[RATE_PER_MILE]))

    saved = save_model(model, gazetteer, bin_edges, FINAL_FEATURES, "hist_gradient_boosting")
    print(f"saved model to {saved}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_validation(model, validation_raw, validation, output_dir / "validation_predictions.csv")
    _write_december(model, december_raw, december, output_dir / "december_predictions.csv")


def _predict(model, frame: pd.DataFrame) -> np.ndarray:
    """Predict `posted_rate` in dollars from the log per-mile model."""
    return np.exp(model.predict(frame[FINAL_FEATURES])) * frame["distance"].to_numpy()


def _write_validation(model, raw: pd.DataFrame, features: pd.DataFrame, path: Path) -> None:
    """Write load_id,predicted_rate for every validation load."""
    output = pd.DataFrame({"load_id": raw["load_id"], "predicted_rate": _predict(model, features)})
    output.to_csv(path, index=False)
    print(f"wrote {path} ({len(output):,} rows)")


def _write_december(model, raw: pd.DataFrame, features: pd.DataFrame, path: Path) -> None:
    """Write the December file with its predicted_rate filled, seven columns kept."""
    output = raw.copy()
    output["predicted_rate"] = _predict(model, features)
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output[DECEMBER_COLUMNS].to_csv(path, index=False)
    print(f"wrote {path} ({len(output):,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and write the submission files.")
    parser.add_argument("--data-dir", default="data/input", type=Path)
    parser.add_argument("--output-dir", default="data/output", type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
