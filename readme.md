# Freight Rate Prediction

Predict the price (`posted_rate`) of US truckload shipments. We train on ~48,000 labelled
loads from Jan–Oct 2025 and predict 12,000 loads from Nov–Dec 2025.

The train/predict split is by time, so this is a **forecasting** problem, not a plain
regression — which is why every choice below is validated on time-ordered folds, never a
random split.

## Setup

Requires **Python 3.12**.

```bash
git clone <repo-url>
cd Spotter-labs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

The project has three entry points, in order.

**1. Search and track** — run every feature/model/hyperparameter search and log it to MLflow:

```bash
python -m src.experiments
```

**2. View the results** — open the MLflow UI to compare every run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**3. Produce the submission** — train the final model and write the prediction files:

```bash
python -m src.pipeline
```

This writes `data/output/validation_predictions.csv`, `data/output/december_predictions.csv`,
and saves the model to `models/`.

**4. Validate and chart** — run the scorer:

```bash
python src/score.py \
  --predictions data/output/validation_predictions.csv \
  --december-predictions data/output/december_predictions.csv
```

It checks the file format and creates `scorer_results/candidate_december.png`.

## Approach

The techniques used, and why:

1. **Temporal validation.** Expanding-window folds (train on the past, test the next block).

A random split would leak the future and inflate the score.
2. **Target transform.** We model `log(posted_rate / distance)`. Dividing by distance removes
the effect that dominates everything else; the log corrects the skew. Predictions are
converted back with `exp(...) * distance`.
3. **Feature engineering.** Cyclical encoding for time , coordinates instead of city names , ratios and distance bins.
4. **Model-based selection.** Features and model are chosen by prediction on the folds
(forward selection, group search, model comparison), not by correlation.
5. **Gradient boosting.** `HistGradientBoostingRegressor` won the model comparison.
6. **Tracking & versioning.** MLflow logs every search; DVC versions the data and models.

## Final model

Fixed from the sweeps in MLflow, then frozen in the code:

- **Model:** `HistGradientBoostingRegressor`, the winner of the model comparison.
- **Hyperparameters:** the best combination from the tuning sweep (a shallow model:
`max_leaf_nodes=15`, `learning_rate=0.03`, `max_iter=200`).
- **Features:** the set chosen by forward selection, plus a design rule on time features.  
The greedy search kept only `dow_cos`, but cyclical time is a **sin/cos pair** — the  
cosine alone is ambiguous. So we keep full pairs: `dow_sin`/`dow_cos` for the weekly  
cycle, and `doy_sin`/`doy_cos` for the yearly cycle. Seasonality did not improve the  
score, but we keep `doy_*` in case it helps on future data with more history.



## Notebook

`notebooks/exploration.ipynb` is used **only for analysis**.

## Project structure

```
src/
  features.py     build the features (cleaning, cyclical time, coordinates, bins)
  validation.py   temporal folds + feature selection (ablation, group search, forward)
  model.py        the model + model comparison + hyperparameter tuning
  tracking.py     MLflow logging helper
  registry.py     save / load the trained model (predict elsewhere)
  experiments.py  run every search and log it to MLflow
  pipeline.py     train the final model and write the predictions   <- run this
  score.py        the scorer (given): validates outputs, makes the chart
notebooks/
  exploration.ipynb   analysis and feature-engineering study
data/
  input/          the given files
  output/         the produced predictions
models/           the saved model (versioned with DVC)
```

## Versioning (DVC)

Code is versioned with git; the **data and the trained model are versioned with DVC**, so
git stays light (it only tracks small `.dvc` pointer files, not the large files).

- `data/input.dvc`, `models.dvc` — pointer files (committed to git)
- `.dvc/cache/` — the actual content, addressed by hash (ignored by git)

After cloning, restore the data and model that match the current commit:

```bash
dvc checkout
```

When the data or model changes, record the new version:

```bash
dvc add data/input models
git add data/input.dvc models.dvc
git commit -m "update data/model"
```

The cache is local. In a team you would add a remote (`dvc remote add ...`) and `dvc push` /
`dvc pull` to share it — out of scope here.

