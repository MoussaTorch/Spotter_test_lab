"""Save and load the trained model with everything needed to predict elsewhere.

The bundle holds the fitted pipeline plus the two lookups learned on the training
data (gazetteer, distance-band edges), so a fresh process can rebuild features
identically and predict from raw data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import features as F

BEST_MODEL_DIR = Path("models")


def save_model(
    pipeline,
    gazetteer: pd.DataFrame,
    bin_edges: list[float],
    feature_list: list[str],
    model_name: str,
    cv_mape: float | None = None,
    directory: Path = BEST_MODEL_DIR,
) -> Path:
    """Persist the fitted pipeline and its provenance under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "gazetteer": gazetteer,
         "bin_edges": bin_edges, "features": feature_list},
        directory / "model.joblib",
    )
    metadata = {
        "model_name": model_name,
        "n_features": len(feature_list),
        "features": feature_list,
        "cv_mape": None if cv_mape is None else round(cv_mape, 3),
        "bin_edges": ["inf" if not np.isfinite(e) else e for e in bin_edges],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def load_model(directory: Path = BEST_MODEL_DIR) -> dict:
    """Load the saved bundle (pipeline + gazetteer + bin_edges + features)."""
    path = directory / "model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"no saved model at {path} — run the pipeline first")
    return joblib.load(path)


def predict_raw(raw: pd.DataFrame, directory: Path = BEST_MODEL_DIR) -> np.ndarray:
    """Predict `posted_rate` from a raw challenge frame using the saved model.

    This is the "test elsewhere" path: it rebuilds features with the saved lookups,
    then predicts — no training, no access to the original notebook needed.
    """
    bundle = load_model(directory)
    frame = F.build_features(raw, gazetteer=bundle["gazetteer"], bin_edges=bundle["bin_edges"])
    per_mile = np.exp(bundle["pipeline"].predict(frame[bundle["features"]]))
    return per_mile * frame["distance"].to_numpy()
