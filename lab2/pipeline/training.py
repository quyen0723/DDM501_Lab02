"""
Model Training Stage for ML Pipeline.

This module handles:
- Model initialization
- Model training
- MLflow experiment tracking
"""

import logging
import pickle
from typing import Any, Dict, Tuple, Optional

import mlflow
from surprise import SVD, NMF, KNNBasic

from pipeline.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_CONFIGS,
    MODELS_DIR,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Model Classes Registry
# =============================================================================
MODEL_CLASSES = {
    "svd": SVD,
    "nmf": NMF,
    "knn": KNNBasic,
}


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
) -> None:
    """
    Setup MLflow tracking.

    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured: URI={tracking_uri}, Experiment={experiment_name}")


# =============================================================================
# TODO 1: train_model
# =============================================================================
def train_model(
    trainset: Any,
    model_type: str = "svd",
    run_name: Optional[str] = None,
    **model_params
) -> Tuple[Any, str]:
    """
    Train a recommendation model and log to MLflow.

    Args:
        trainset: Surprise trainset object
        model_type: Type of model ('svd', 'nmf', 'knn')
        run_name: Optional name for the MLflow run
        **model_params: Model hyperparameters

    Returns:
        Tuple of (trained_model, run_id)

    Example:
        model, run_id = train_model(
            trainset,
            model_type='svd',
            n_factors=100,
            n_epochs=20
        )
    """
    model_class = get_model_class(model_type)

    with mlflow.start_run(run_name=run_name):
        # ---- Log parameters -------------------------------------------------
        mlflow.log_param("model_type", model_type)
        for key, value in model_params.items():
            mlflow.log_param(key, value)

        # Dataset info (useful for reproducibility)
        mlflow.log_param("n_train_ratings", trainset.n_ratings)
        mlflow.log_param("n_users", trainset.n_users)
        mlflow.log_param("n_items", trainset.n_items)

        # ---- Initialize & train model ---------------------------------------
        model = model_class(**model_params)

        logger.info(f"Training {model_type} model with params: {model_params}")
        model.fit(trainset)

        # ---- Save & log model artifact --------------------------------------
        model_path = MODELS_DIR / f"model_{model_type}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Training complete. Run ID: {run_id}")

        return model, run_id


# =============================================================================
# TODO 2: train_with_config
# =============================================================================
def train_with_config(trainset: Any, config: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Train model using a configuration dictionary.

    Args:
        trainset: Surprise trainset object
        config: Configuration dictionary with model_type and hyperparameters

    Returns:
        Tuple of (trained_model, run_id)

    Example:
        config = {"model_type": "svd", "n_factors": 100, "n_epochs": 20}
        model, run_id = train_with_config(trainset, config)
    """
    config_copy = config.copy()
    model_type = config_copy.pop("model_type")
    return train_model(trainset, model_type=model_type, **config_copy)


# =============================================================================
# TODO 3: get_model_class (BONUS)
# =============================================================================
def get_model_class(model_type: str):
    """
    Get the model class for a given model type.

    Args:
        model_type: Type of model ('svd', 'nmf', 'knn')

    Returns:
        Model class from Surprise library

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: {list(MODEL_CLASSES.keys())}"
        )
    return MODEL_CLASSES[model_type]


# =============================================================================
# Helper functions (PROVIDED)
# =============================================================================
def get_default_params(model_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a model type.

    Args:
        model_type: Type of model

    Returns:
        Dictionary of default parameters
    """
    return MODEL_CONFIGS.get(model_type, {})


def list_available_models() -> list:
    """
    List all available model types.

    Returns:
        List of model type names
    """
    return list(MODEL_CLASSES.keys())


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    from pipeline.data_ingestion import load_and_split

    print("Testing Training Module")
    print("=" * 50)

    # Setup MLflow
    setup_mlflow()

    # Load data
    trainset, testset, _ = load_and_split()

    # Test training
    model, run_id = train_model(
        trainset,
        model_type="svd",
        run_name="test_run",
        n_factors=50,
        n_epochs=10
    )
    print(f"Model trained. Run ID: {run_id}")

    print("\nAvailable models:", list_available_models())
    print("Default SVD params:", get_default_params("svd"))
