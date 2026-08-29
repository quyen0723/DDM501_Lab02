# MLflow & Airflow UI Screenshots — Lab 2

Submission evidence for **§5.3 — "Screenshots of the MLflow UI showing
experiments"** and the **Experiment Tracking / Airflow Automation** rubric
criteria (§5.2).

`screenshots/*.png` is allow-listed in `.gitignore`, so the PNGs below are
tracked in the repository.

## Screenshots

| File | Source | What it shows | Rubric (§5.2) |
|---|---|---|---|
| `experiment_comparison.png` | MLflow UI → experiment `movie-rating-prediction` → **Compare** view | Runs sorted by `rmse`; parameters (model_type, n_epochs, n_factors, …) vs metrics (rmse, mae, mse, mape, coverage) | Experiment Tracking 25% (params + metrics logged) — **§5.3 required** |
| `experiments_overview.png` | MLflow UI → Experiments list | Experiment `movie-rating-prediction` (ID 1) run table (Run Name, Created, Duration, Source, Models, Metrics) | Experiment Tracking 25% (MLflow setup + runs logged) |
| `airflow_dags_list.png` | Airflow UI (`http://localhost:8080`) → DAGs | DAG `movie_rating_training` loaded and active | Airflow Automation 20% (DAG structure) |

> **Model Registry** (rubric 3%) is implemented in `pipeline/registry.py`
> (`register_best_model`) and verified live on the MLflow server:
> `movie-rating-model` is registered with a version in the **Production**
> stage. A Models-tab screenshot is optional — §5.3 only requires experiment
> screenshots, which are provided above.

## How these screenshots were produced

```bash
# 1. Bring up the MLflow + Airflow stack (Docker Compose v2 plugin).
docker compose up -d
#   MLflow UI  -> http://localhost:5000
#   Airflow UI -> http://localhost:8080  (admin / admin)

# 2. Run the 9 hyperparameter experiments against the MLflow server.
#    The MovieLens download prompt is answered non-interactively (see repo README).
export MLFLOW_TRACKING_URI=http://localhost:5000
echo "Y" | python -m experiments.run_experiments

# 3. (Optional) Register the best run in the Model Registry.
python -c "from pipeline.registry import register_best_model; register_best_model()"

# 4. (Or via Airflow) enable the `movie_rating_training` DAG in the Airflow UI
#    and trigger it manually.
```

Then open `http://localhost:5000`, select experiment `movie-rating-prediction`,
tick the runs and click **Compare** (sort by `rmse`) → capture
`experiment_comparison.png`.

## Notes

- The experiment name is `movie-rating-prediction`, defined by
  `MLFLOW_EXPERIMENT_NAME` in `pipeline/config.py`. `register_best_model()`
  defaults to the same name, so no argument is required.
- `experiment_report.md` (in `lab2/`) contains the full 9-experiment report
  with analysis — the deliverable for the Documentation rubric (Experiment
  report 10%).
- Use `docker compose` (space, Compose v2 plugin), not `docker-compose` (v1 is
  not installed on this host).