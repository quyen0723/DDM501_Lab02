# DDM501 Lab 2 — ML Pipeline & Experiment Tracking

> The actual project lives in [`lab2/`](./lab2) — open [`lab2/README.md`](./lab2/README.md) for the full documentation.

## What this is

A reproducible **ML pipeline** for the movie-rating prediction system from Lab 1, with
**MLflow** experiment tracking + Model Registry and **Apache Airflow** orchestration, for
course **DDM501** (Lab 2, weight 15%).

It adds, on top of the Lab 1 model:

- A modular 5-stage pipeline (ingestion → preprocessing → training → evaluation → registration)
- MLflow tracking of params, metrics (RMSE/MAE/MSE/MAPE/coverage), and artifacts (model + plots)
- Hyperparameter tuning across 3 model families (SVD, NMF, KNN) with 9 experiments + a report
- MLflow Model Registry: best run auto-registered and promoted to Production
- An Airflow DAG that runs the pipeline weekly and registers the model only if RMSE < 1.0
- A full Docker Compose stack (MLflow server + Postgres + Airflow webserver/scheduler + bonus pipeline service)

## Repository layout

```
DDM501_Lab02/
├── README.md            ← you are here (project landing page)
└── lab2/                ← the deliverable
    ├── pipeline/        # config, ingestion, preprocessing, training, evaluation, registry, run_pipeline
    ├── dags/            # Airflow training DAG (7 tasks, RMSE-gated registration)
    ├── experiments/     # hyperparameter tuning runner + report generator
    ├── tests/           # pytest suite (11 tests) + conftest.py (self-contained)
    ├── scripts/         # setup_mlflow.py
    ├── screenshots/     # MLflow UI screenshots (add before submitting)
    ├── Dockerfile       # bonus ml-pipeline image (scikit-surprise build fix)
    ├── Dockerfile.airflow# custom Airflow image (surprise + mlflow + MovieLens baked in)
    ├── docker-compose.yml
    ├── experiment_report.md  # 9-experiment report with analysis (deliverable)
    └── requirements*.txt
```

## Deliverables vs. grading rubric

| Rubric (weight) | Where |
|-----------------|-------|
| Pipeline Quality (35%) — modular, reproducible, error handling, code quality | `lab2/pipeline/`, `lab2/pipeline/run_pipeline.py` |
| Experiment Tracking (25%) — MLflow setup, params, metrics, artifacts, registry | `lab2/pipeline/training.py`, `evaluation.py`, `registry.py` |
| Airflow Automation (20%) — DAG, tasks execute, schedule | `lab2/dags/ml_training_dag.py` |
| Documentation (20%) — experiment report, README, code docs | `lab2/experiment_report.md`, `lab2/README.md`, docstrings |

## Quick start

```bash
cd lab2

# 1. Install (scikit-surprise 1.1.3 needs build prerequisites — see lab2/README.md)
python -m venv venv && source venv/bin/activate
pip install "cython<3" "numpy==1.26.2" "setuptools<81"
pip install --no-build-isolation -r requirements-pipeline.txt

# 2. Run the pipeline (local file tracking, no server needed)
export MLFLOW_TRACKING_URI=file:./mlruns
python -m pipeline.run_pipeline

# 3. Run all 9 hyperparameter experiments + generate the report
python -m experiments.run_experiments   # → experiment_report.md

# 4. Register the best model in the Model Registry
python -c "from pipeline.registry import register_best_model; \
          register_best_model(experiment_name='hyperparameter-tuning')"

# 5. Full stack (MLflow UI :5000 + Airflow UI :8080 admin/admin)
docker compose up -d --build

# 6. Tests (self-contained — no MLflow server required)
pytest tests/ -v
```

## Team — Group 1 (DDM501 — Lab 2)

| # | Member |
|---|--------|
| 1 | Nguyễn Thị Hồng Ngọc |
| 2 | Trần Huỳnh Thanh Trúc |
| 3 | Nguyễn Tất Hiển |
| 4 | Nguyễn Ngọc Mỹ Quyên |
| 5 | Huỳnh Thị Thanh Vi |