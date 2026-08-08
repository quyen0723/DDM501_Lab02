# Lab 2: ML Pipeline & Experiment Tracking

Reproducible ML pipeline for movie rating prediction with **MLflow** experiment tracking, **Model Registry**, and **Airflow** orchestration.

## Features

- Modular 5-stage pipeline: ingestion → preprocessing → training → evaluation → registration
- Supports 3 model families: **SVD**, **NMF**, **KNN** (scikit-surprise)
- Every run logs params, metrics (RMSE, MAE, MSE, MAPE, coverage), and artifacts (model .pkl + evaluation plots) to MLflow
- Hyperparameter tuning runner with 9 predefined configs + auto-generated markdown report
- Best model auto-registered to MLflow Model Registry and promoted to Production
- Airflow DAG with branching: only registers the model if RMSE < 1.0
- Full Docker Compose stack: MLflow server + Postgres + Airflow webserver/scheduler + bonus pipeline service

## Project Structure

```
ddm501-lab2/
├── pipeline/
│   ├── config.py           # Paths, MLflow, model & experiment configs
│   ├── data_ingestion.py   # Load MovieLens 100K, train/test split
│   ├── preprocessing.py    # Validation & data statistics
│   ├── training.py         # train_model() with MLflow logging
│   ├── evaluation.py       # evaluate_model() with metrics + plots
│   ├── registry.py         # find_best_run / register / promote
│   └── run_pipeline.py     # CLI orchestrator for all stages
├── dags/
│   └── ml_training_dag.py  # Airflow DAG (7 tasks, with branch)
├── experiments/
│   └── run_experiments.py  # Hyperparameter tuning + report
├── tests/
│   ├── test_pipeline.py    # Unit tests (11 tests)
│   └── conftest.py         # Self-contained tests (local MLflow + MovieLens)
├── scripts/
│   └── setup_mlflow.py     # MLflow setup & smoke test
├── screenshots/            # MLflow UI screenshots (lab deliverable) — add before submit
├── docker-compose.yml      # MLflow + Airflow (custom image) + bonus pipeline service
├── Dockerfile              # Image for the bonus ml-pipeline service (surprise build fix)
├── Dockerfile.airflow      # Custom Airflow image with surprise + mlflow + MovieLens baked in
├── requirements.txt
└── requirements-pipeline.txt
```

## Quick Start

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate
# scikit-surprise 1.1.3 build prerequisites (see Troubleshooting for why)
pip install "cython<3" "numpy==1.26.2" "setuptools<81"
pip install --no-build-isolation -r requirements.txt
```

> Tip: if you only want the pipeline (not Airflow), the slimmer
> `requirements-pipeline.txt` is enough.

### 2. Start MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```

MLflow UI: http://localhost:5000

> No server? The pipeline also works with local file tracking:
> `export MLFLOW_TRACKING_URI=file:./mlruns`

### 3. Run the Pipeline

```bash
# Full pipeline with defaults (SVD, 100 factors, 20 epochs)
python -m pipeline.run_pipeline

# Custom hyperparameters + register best model to Production
python -m pipeline.run_pipeline --model-type svd --n-factors 100 --n-epochs 20 --register
```

### 4. Run Hyperparameter Experiments

```bash
python -m experiments.run_experiments
```

Runs all 9 configs in `pipeline/config.py` (4×SVD, 2×NMF, 3×KNN), logs everything to the `hyperparameter-tuning` experiment, and writes `experiment_report.md` with a results table, best model, and recommendations. Compare runs side-by-side in the MLflow UI.

### 5. Airflow (Docker)

```bash
docker-compose up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8080 | admin / admin |

Enable the `movie_rating_training` DAG in the Airflow UI. It runs weekly (`@weekly`) with the flow:

```
load_data → preprocess_data → train_model → evaluate_model → decide_registration
                                                   ├─ register_model ─┐
                                                   └─ skip_registration ┴→ cleanup
```

The branch registers the model **only if RMSE < 1.0**; `cleanup` runs either way (`trigger_rule='none_failed'`).

Bonus: run the pipeline once inside Docker against the MLflow server:

```bash
docker-compose run --rm ml-pipeline
```

## MLflow Tracking Details

Per run, the pipeline logs:

| Type | Items |
|------|-------|
| Params | model_type, all hyperparameters, dataset size (n_users/n_items/n_ratings) |
| Metrics | rmse, mae, mse, mape, coverage |
| Artifacts | model pickle (`model/`), prediction_distribution.png, error_by_rating.png |

Model Registry: `register_best_model()` finds the lowest-RMSE run, registers it as `movie-rating-model`, and transitions it to Production (archiving previous Production versions).

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=pipeline
pytest tests/ -v -m "not slow"   # skip slow integration tests
```

## Experiment Report

After `python -m experiments.run_experiments`, see `experiment_report.md` for:
1. Experiment setup and summary
2. Results table (all runs, RMSE/MAE)
3. Best model configuration and run ID
4. Recommendations for production

Attach MLflow UI screenshots (experiment comparison view + registered model page) to the submission.

## Troubleshooting

- **`pip install` fails on scikit-surprise / `import surprise` fails with `No module named 'pkg_resources'`**:
  scikit-surprise 1.1.3 predates Cython 3 and numpy 2, and imports the `pkg_resources`
  module that setuptools ≥ 81 removed. The requirements files already pin
  `cython<3`, `numpy==1.26.2`, and `setuptools<81`. When installing **outside
  Docker on a fresh venv**, install the build prerequisites first, then surprise
  with `--no-build-isolation` so it compiles against the pinned numpy:

  ```bash
  pip install "cython<3" "numpy==1.26.2" "setuptools<81"
  pip install --no-build-isolation -r requirements-pipeline.txt   # or requirements.txt
  ```

  The `Dockerfile` (and `Dockerfile.airflow`) do this for you.
- **`mlflow` CLI errors on Python 3.12** (`'EntryPoints' object has no attribute 'get'`):
  a known mlflow 2.9.2 ↔ Python 3.12 CLI incompatibility. Run the MLflow **server in
  Docker** (`docker-compose up -d` → http://localhost:5000) — the prebuilt
  `ghcr.io/mlflow/mlflow:v2.9.2` image is unaffected — or use local file tracking
  (`export MLFLOW_TRACKING_URI=file:./mlruns`) which uses the Python API and works
  everywhere.
- **MLflow "database is locked"**: SQLite backend is single-writer; don't run many parallel pipelines against it.
- **`transition_model_version_stage` deprecation warning**: expected on MLflow 2.9 — stages still work; aliases are the successor API.

## Deliverables checklist (Lab 2)

- [x] Modular ML pipeline (`pipeline/`) + CLI orchestrator (`run_pipeline.py`)
- [x] MLflow tracking — params, metrics (RMSE/MAE/MSE/MAPE/coverage), artifacts (model + plots), Model Registry
- [x] Airflow DAG (`dags/ml_training_dag.py`, weekly, RMSE-gated registration)
- [x] Experiment report — [`experiment_report.md`](./experiment_report.md) (9 experiments: 4×SVD, 2×NMF, 3×KNN, with analysis)
- [x] README (this file) + docstrings on all TODO functions
- [ ] **MLflow UI screenshots** in [`screenshots/`](./screenshots) — capture before submitting (see that folder's README)

## Team — Group 1 (DDM501 — Lab 2)

| # | Member |
|---|--------|
| 1 | Nguyễn Thị Hồng Ngọc |
| 2 | Trần Huỳnh Thanh Trúc |
| 3 | Nguyễn Tất Hiển |
| 4 | Nguyễn Ngọc Mỹ Quyên |
| 5 | Huỳnh Thị Thanh Vi |
