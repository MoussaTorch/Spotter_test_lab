"""Model, training, prediction, and search.

Final model: histogram gradient boosting, trained on `log(rate_per_mile)`.
The choice is not guessed — `compare_models` and `tune_hyperparameters` decide it.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL, RATE_PER_MILE
from .tracking import track
from .validation import evaluate_features, summarise

# Best hyperparameters from the tuning sweep. See MLflow.
DEFAULT_HGB_PARAMS: dict[str, object] = {
    "max_iter": 200,
    "learning_rate": 0.03,
    "max_leaf_nodes": 15,
    "random_state": 0,
}

# Grid searched by `tune_hyperparameters` when none is supplied.
DEFAULT_PARAM_GRID: dict[str, list[object]] = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [200, 400, 600],
    "max_leaf_nodes": [15, 31, 63],
}


def _pipeline(features: Sequence[str], estimator: BaseEstimator) -> Pipeline:
    """Wrap an estimator behind one-hot encoding of the categoricals."""
    categorical = [column for column in features if column in CATEGORICAL]
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", preprocessor), ("regressor", estimator)])


def make_model(features: Sequence[str], **hyperparams: object) -> Pipeline:
    """Build the frozen final model: the gradient-boosting pipeline.

    Gradient boosting is not assumed — it is the family that won `compare_models`
    on the temporal folds (reproduce with `select_best_model`). It is frozen here
    so the prediction scripts are deterministic; `train_selected` is the dynamic
    path that re-derives the winner instead.
    """
    params = {**DEFAULT_HGB_PARAMS, **hyperparams}
    return _pipeline(features, HistGradientBoostingRegressor(**params))


def train(frame: pd.DataFrame, features: Sequence[str], **hyperparams: object) -> Pipeline:
    """Fit the pipeline on the log per-mile target."""
    model = make_model(features, **hyperparams)
    model.fit(frame[list(features)], np.log(frame[RATE_PER_MILE]))
    return model


def predict(model: Pipeline, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Predict `posted_rate` in dollars, undoing the log-per-mile transform."""
    per_mile = np.exp(model.predict(frame[list(features)]))
    return per_mile * frame["distance"].to_numpy()


def default_candidates() -> dict[str, BaseEstimator]:
    """The model families compared: one linear, two tree ensembles."""
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(**DEFAULT_HGB_PARAMS),
    }


def compare_models(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    candidates: Mapping[str, BaseEstimator] | None = None,
    experiment: str | None = None,
) -> pd.DataFrame:
    """Score several model families on the temporal folds, best MAPE first.

    Same features and folds for all — only the algorithm changes. Pass
    `experiment` to log each family as an MLflow run.
    """
    candidates = candidates or default_candidates()
    rows: dict[str, dict[str, float]] = {}
    for name, estimator in candidates.items():
        factory = lambda columns, est=estimator: _pipeline(columns, clone(est))
        summary = summarise(evaluate_features(frame, features, factory, folds))
        with track(experiment, name, {"model": name, "n_features": len(features)}, summary):
            rows[name] = summary
    return pd.DataFrame(rows).T.sort_values("mape").round(3)


def select_best_model(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    candidates: Mapping[str, BaseEstimator] | None = None,
) -> tuple[str, BaseEstimator]:
    """Pick the winning family from `compare_models`: the choice is derived, not asserted."""
    candidates = dict(candidates) if candidates is not None else default_candidates()
    ranking = compare_models(frame, features, folds, candidates)
    best_name = str(ranking.index[0])
    return best_name, clone(candidates[best_name])


def train_selected(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    candidates: Mapping[str, BaseEstimator] | None = None,
) -> tuple[str, Pipeline]:
    """Dynamic path: fit whatever family won the comparison on the whole frame."""
    name, estimator = select_best_model(frame, features, folds, candidates)
    pipeline = _pipeline(features, estimator)
    pipeline.fit(frame[list(features)], np.log(frame[RATE_PER_MILE]))
    return name, pipeline


def hyperparameter_combinations(param_grid: Mapping[str, Iterable[object]]) -> Iterator[dict[str, object]]:
    """Yield every combination in the grid (the Cartesian product of the values)."""
    keys = list(param_grid)
    for values in itertools.product(*(list(param_grid[key]) for key in keys)):
        yield dict(zip(keys, values))


def tune_hyperparameters(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    param_grid: Mapping[str, Iterable[object]] | None = None,
    experiment: str | None = None,
) -> pd.DataFrame:
    """Grid-search the hyperparameters over the temporal folds, best MAPE first.

    A flat score across a value means it is not worth tuning; a moving score means
    it matters. Pass `experiment` to log each combination as an MLflow run.
    """
    param_grid = param_grid or DEFAULT_PARAM_GRID
    rows: list[dict[str, object]] = []
    for combination in hyperparameter_combinations(param_grid):
        factory = lambda columns, combo=combination: make_model(columns, **combo)
        scores = evaluate_features(frame, features, factory, folds)
        metrics = {"mape": float(scores.mean()), "std": float(scores.std(ddof=1))}
        run_name = ", ".join(f"{k}={v}" for k, v in combination.items())
        with track(experiment, run_name, dict(combination), metrics):
            rows.append({**combination, **metrics})
    return pd.DataFrame(rows).sort_values("mape").round(3).reset_index(drop=True)
