# MLflow UI Screenshots

The assignment (Lab 2, §5.3) explicitly requires **screenshots of the MLflow UI
showing experiments**. Capture them and drop the PNG files into this folder, then
commit — `screenshots/*.png` is allow-listed in `.gitignore` so it is tracked.

## Required screenshots

1. **Experiment comparison view**
   - Start the stack: `docker-compose up -d`
   - Open http://localhost:5000
   - Select the `hyperparameter-tuning` experiment
   - Use the "Compare" view to sort runs by `rmse` and show params vs metrics
   - Save as `experiment_comparison.png`

2. **Registered model page**
   - In the MLflow UI → "Models" tab → `movie-rating-model`
   - Show version 1 in the "Production" stage
   - Save as `registered_model.png`

## How to produce the runs the screenshots show

```bash
# 1. Bring up MLflow + Airflow
docker-compose up -d

# 2. Run the 9 hyperparameter experiments against the MLflow server
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m experiments.run_experiments

# 3. Register the best run in the Model Registry
python -c "from pipeline.registry import register_best_model; \
          register_best_model(experiment_name='hyperparameter-tuning')"

# 4. (Or via Airflow) enable the `movie_rating_training` DAG in the Airflow UI
#    at http://localhost:8080 (admin / admin) and trigger it manually.
```

Then capture the two screenshots above and commit them:

```bash
git add screenshots/experiment_comparison.png screenshots/registered_model.png
git commit -m "docs: add MLflow UI screenshots for Lab 2 submission"
```

> If no screenshots are present here, the Documentation rubric (report / submission
> requirement) is incomplete.