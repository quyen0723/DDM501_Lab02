"""
Model Registry Stage for ML Pipeline.

This module handles:
- Finding the best model from experiments
- Registering models to MLflow Model Registry
- Managing model versions and stages
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import MLFLOW_EXPERIMENT_NAME

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TODO 1: find_best_run
# =============================================================================
def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    ascending: bool = True
) -> Dict[str, Any]:
    """
    Find the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to optimize (default: 'rmse')
        ascending: If True, lower is better (default: True for RMSE)

    Returns:
        Dictionary with best run information:
        {
            'run_id': str,
            'metrics': dict,
            'params': dict,
            'artifact_uri': str
        }

    Example:
        best = find_best_run(metric='rmse', ascending=True)
        print(f"Best RMSE: {best['metrics']['rmse']}")
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} > 0",  # only runs that logged the metric
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs with metric '{metric}' found in experiment '{experiment_name}'")

    best_run = runs[0]
    logger.info(
        f"Best run: {best_run.info.run_id} "
        f"({metric}={best_run.data.metrics.get(metric)})"
    )

    return {
        "run_id": best_run.info.run_id,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "artifact_uri": best_run.info.artifact_uri,
    }


# =============================================================================
# TODO 2: register_model
# =============================================================================
def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model"
) -> str:
    """
    Register a model from an MLflow run to the Model Registry.

    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
        artifact_path: Path to the model artifact within the run

    Returns:
        Version number of the registered model (as string)

    Example:
        version = register_model(run_id, "movie-rating-model")
        print(f"Registered model version: {version}")
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"Registering model from {model_uri} as '{model_name}'")

    result = mlflow.register_model(model_uri, model_name)

    logger.info(f"Model registered: {model_name} version {result.version}")
    return result.version


# =============================================================================
# TODO 3: transition_model_stage
# =============================================================================
def transition_model_stage(
    model_name: str,
    version: str,
    stage: str = "Production"
) -> None:
    """
    Transition a model version to a new stage.

    Args:
        model_name: Name of the registered model
        version: Version number to transition
        stage: Target stage ('Staging', 'Production', 'Archived')

    Valid stages:
        - 'None': No stage
        - 'Staging': For testing
        - 'Production': For production use
        - 'Archived': Retired models

    Example:
        transition_model_stage("movie-rating-model", "1", "Production")
    """
    valid_stages = {"None", "Staging", "Production", "Archived"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Valid stages: {valid_stages}")

    client = MlflowClient()

    logger.info(f"Transitioning {model_name} v{version} to {stage}")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )

    logger.info(f"Model {model_name} v{version} is now in {stage}")


# =============================================================================
# TODO 4: register_best_model (combines all above)
# =============================================================================
def register_best_model(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    model_name: str = "movie-rating-model",
    metric: str = "rmse",
    stage: str = "Production"
) -> Dict[str, Any]:
    """
    Find the best model and register it to the Model Registry.

    Args:
        experiment_name: Name of the MLflow experiment
        model_name: Name for the registered model
        metric: Metric to optimize (default: 'rmse')
        stage: Stage to transition to (default: 'Production')

    Returns:
        Dictionary with registration info:
        {
            'run_id': str,
            'model_name': str,
            'version': str,
            'stage': str,
            'metrics': dict
        }

    Example:
        result = register_best_model()
        print(f"Registered {result['model_name']} v{result['version']}")
    """
    # 1. Find best run
    best_run = find_best_run(experiment_name, metric, ascending=True)
    logger.info(
        f"Best run: {best_run['run_id']} with {metric}={best_run['metrics'].get(metric)}"
    )

    # 2. Register model
    version = register_model(best_run["run_id"], model_name)

    # 3. Transition to stage
    transition_model_stage(model_name, version, stage)

    return {
        "run_id": best_run["run_id"],
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "metrics": best_run["metrics"],
    }


# =============================================================================
# Helper Functions (PROVIDED)
# =============================================================================
def list_registered_models() -> List[Dict[str, Any]]:
    """
    List all registered models.

    Returns:
        List of model information dictionaries
    """
    client = MlflowClient()
    models = client.search_registered_models()

    return [
        {
            "name": model.name,
            "latest_versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                }
                for v in model.latest_versions
            ]
        }
        for model in models
    ]


def get_production_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the current production version of a model.

    Args:
        model_name: Name of the registered model

    Returns:
        Dictionary with model info or None if not found
    """
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            v = versions[0]
            return {
                "name": model_name,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
            }
    except Exception as e:
        logger.error(f"Error getting production model: {e}")

    return None


def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Get top N runs from an experiment.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        top_n: Number of runs to return

    Returns:
        List of run information sorted by metric
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=top_n
    )

    return [
        {
            "run_id": run.info.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
        for run in runs
    ]


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    print("Testing Registry Module")
    print("=" * 50)

    # Test helper functions
    print("\nRegistered models:", list_registered_models())

    # After running experiments, test:
    # result = register_best_model()
    # print(f"Registered: {result}")

    print("\nRegistry module loaded successfully.")
