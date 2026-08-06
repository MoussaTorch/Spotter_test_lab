"""Run every search with MLflow tracking.

Run this to populate MLflow with all the evidence behind the final model:
    python -m src.experiments
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

from pathlib import Path

from . import features as F
from .model import compare_models, make_model, tune_hyperparameters, DEFAULT_PARAM_GRID
from .validation import temporal_folds, search_feature_groups, forward_selection


# Candidate features offered to forward selection (categoricals + the numerics).
SELECTION_CANDIDATES = [
    "distance", "equipment", "weight_per_mile", "distance_band", "weight",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos",
]


def run_all(data_dir: Path = Path("data/input")) -> None:
    """Execute all searches, logging each to its own MLflow experiment."""
    train = F.add_target(F.build_features(F.load_dataset(data_dir / "train-test.csv")))
    folds = temporal_folds(train, "2025-07-01", "2025-10-18")
    features = F.feature_names()

    print("1/4 feature-group search ...")
    search_feature_groups(train, make_model, folds,
                          required=["shipment"], experiment="01_feature_group_search")

    print("2/4 forward selection ...")
    forward_selection(train, SELECTION_CANDIDATES, make_model, folds,
                      experiment="02_feature_forward_selection")

    print("3/4 model comparison ...")
    compare_models(train, features, folds, experiment="03_model_comparison")

    print("4/4 hyperparameter tuning (full grid) ...")
    tune_hyperparameters(train, features, folds, param_grid=DEFAULT_PARAM_GRID,
                         experiment="04_hyperparameter_tuning")

    print("done — view with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    run_all()
