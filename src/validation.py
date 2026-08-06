"""Time-aware validation and feature selection.

The target is in the future of the training data, so scores come from
expanding-window temporal folds — the model only ever sees the past. A random
split would leak the future and flatter the scores.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import FEATURE_GROUPS, RATE_PER_MILE, TARGET
from .tracking import track

# A model factory: called with the feature list, returns a fresh scikit-learn
# estimator. It gets the features so the pipeline only encodes the categoricals present.
ModelFactory = Callable[[Sequence[str]], object]


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, the challenge's relative metric."""
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def temporal_folds(
    frame: pd.DataFrame,
    start: str,
    end: str,
    horizon_days: int = 14,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds: train before a cut, test the next block.

    The cut advances by `horizon_days`. Many folds tell a real gain from noise.
    """
    cuts = pd.date_range(start, end, freq=f"{horizon_days}D")
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for cut in cuts:
        train = frame[frame["date"] < cut]
        test = frame[(frame["date"] >= cut) & (frame["date"] < cut + pd.Timedelta(days=horizon_days))]
        if len(test) >= 500:
            folds.append((train, test))
    if not folds:
        raise ValueError("no usable folds — check the date range")
    return folds


def evaluate_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    make_model: ModelFactory,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> np.ndarray:
    """MAPE of one feature set across the folds.

    Trained on `log(rate_per_mile)`, scored back in dollars with `exp(...) * distance`.
    """
    columns = list(features)
    scores = []
    for train, test in folds:
        model = make_model(columns)
        model.fit(train[columns], np.log(train[RATE_PER_MILE]))
        predicted = np.exp(model.predict(test[columns])) * test["distance"]
        scores.append(mape(test[TARGET].to_numpy(), predicted))
    return np.asarray(scores)


def summarise(scores: np.ndarray) -> dict[str, float]:
    """Mean, spread and standard error of a fold score vector."""
    return {
        "mape": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "stderr": float(scores.std(ddof=1) / np.sqrt(len(scores))),
        "folds": len(scores),
    }


def ablation(
    frame: pd.DataFrame,
    feature_sets: dict[str, Sequence[str]],
    make_model: ModelFactory,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    experiment: str | None = None,
) -> pd.DataFrame:
    """Score several named feature sets, best (lowest MAPE) first.

    Pass `experiment` to log each set as an MLflow run.
    """
    rows: dict[str, dict[str, float]] = {}
    for name, features in feature_sets.items():
        summary = summarise(evaluate_features(frame, features, make_model, folds))
        with track(experiment, name, {"feature_set": name, "n_features": len(features)}, summary):
            rows[name] = summary
    return pd.DataFrame(rows).T.sort_values("mape").round(3)


def search_feature_groups(
    frame: pd.DataFrame,
    make_model: ModelFactory,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    groups: Mapping[str, Sequence[str]] | None = None,
    required: Sequence[str] = (),
    experiment: str | None = None,
) -> pd.DataFrame:
    """Score every combination of feature groups, best MAPE first.

    Searches at the group level (4 groups = 15 combinations). `required` forces
    groups into every combination — where domain constraints live, e.g. keeping a
    time group so the December chart is never flat. Pass `experiment` to log each
    combination as an MLflow run.
    """
    groups = groups or FEATURE_GROUPS
    optional = [name for name in groups if name not in required]

    rows: list[dict[str, object]] = []
    for size in range(len(optional) + 1):
        for extra in itertools.combinations(optional, size):
            names = list(required) + list(extra)
            if not names:
                continue
            features = [column for name in names for column in groups[name]]
            summary = summarise(evaluate_features(frame, features, make_model, folds))
            combo = " + ".join(names)
            with track(experiment, combo, {"groups": combo, "n_features": len(features)}, summary):
                rows.append({"groups": combo, "n_features": len(features), **summary})
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def forward_selection(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    make_model: ModelFactory,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    tolerance: float = 0.0,
    experiment: str | None = None,
) -> pd.DataFrame:
    """Greedily add the feature that most improves MAPE, until none helps.

    Model-based selection: it asks "does this feature help predictions?", not
    "is it correlated?". A feature is kept only if it beats the score by `tolerance`.
    Pass `experiment` to log each accepted step as an MLflow run.
    """
    remaining = list(candidates)
    selected: list[str] = []
    best_score = float("inf")
    history: list[dict[str, object]] = []

    while remaining:
        trials = {
            candidate: evaluate_features(frame, selected + [candidate], make_model, folds).mean()
            for candidate in remaining
        }
        candidate, score = min(trials.items(), key=lambda item: item[1])
        if score >= best_score - tolerance:
            break
        selected.append(candidate)
        remaining.remove(candidate)
        best_score = score
        with track(experiment, f"step_{len(selected):02d}_{candidate}",
                   {"added": candidate, "n_features": len(selected), "features": ",".join(selected)},
                   {"mape": float(score)}):
            history.append({"step": len(selected), "added": candidate, "mape": round(score, 3)})

    return pd.DataFrame(history)
