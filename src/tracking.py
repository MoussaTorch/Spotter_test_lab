"""MLflow tracking for the model and hyperparameter searches.

Every candidate in a search is logged as one MLflow run (its params and fold
metrics), so the whole search is visible and comparable in the MLflow UI:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

TRACKING_URI = "sqlite:///mlflow.db"


@contextmanager
def track(experiment: str | None, run_name: str, params: dict, metrics: dict) -> Iterator[None]:
    """Log one search result as an MLflow run."""
    if experiment is None:
        yield
        return

    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        yield
