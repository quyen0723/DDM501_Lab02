"""
Model Evaluation Stage for ML Pipeline.

This module handles:
- Making predictions on test data
- Calculating evaluation metrics
- Logging metrics to MLflow
- Creating evaluation visualizations
"""

import logging
from typing import Any, Dict, List

import mlflow
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (safe for servers/Airflow)
import matplotlib.pyplot as plt
from surprise import accuracy

from pipeline.config import ARTIFACTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TODO 1: evaluate_model
# =============================================================================
def evaluate_model(
    model: Any,
    testset: List,
    run_id: str,
    log_to_mlflow: bool = True
) -> Dict[str, float]:
    """
    Evaluate model and log metrics to MLflow.

    Args:
        model: Trained Surprise model
        testset: Test set as list of (user, item, rating) tuples
        run_id: MLflow run ID to log metrics to
        log_to_mlflow: Whether to log metrics to MLflow

    Returns:
        Dictionary with evaluation metrics {'rmse': float, 'mae': float, ...}

    Example:
        metrics = evaluate_model(model, testset, run_id)
        print(f"RMSE: {metrics['rmse']:.4f}")
    """
    logger.info("Evaluating model...")

    # 1. Make predictions on the test set
    predictions = model.test(testset)

    # 2. Core metrics
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    metrics = {"rmse": rmse, "mae": mae}

    # 3. Additional metrics
    extra = calculate_additional_metrics(predictions)
    metrics.update({k: v for k, v in extra.items() if v is not None})

    # 4. Log to MLflow (resume the training run)
    if log_to_mlflow:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            if extra.get("mse") is not None:
                mlflow.log_metric("mse", extra["mse"])
            if extra.get("mape") is not None:
                mlflow.log_metric("mape", extra["mape"])
            if extra.get("coverage") is not None:
                mlflow.log_metric("coverage", extra["coverage"])

            # Create and log evaluation plots
            fig = create_prediction_distribution_plot(predictions)
            mlflow.log_figure(fig, "prediction_distribution.png")
            plt.close(fig)

            fig2 = create_error_by_rating_plot(predictions)
            mlflow.log_figure(fig2, "error_by_rating.png")
            plt.close(fig2)

    logger.info(f"Evaluation complete. RMSE={rmse:.4f}, MAE={mae:.4f}")
    return metrics


# =============================================================================
# TODO 2: calculate_additional_metrics
# =============================================================================
def calculate_additional_metrics(predictions: List) -> Dict[str, float]:
    """
    Calculate additional evaluation metrics beyond RMSE and MAE.

    Metrics:
    - MSE: Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (skips zero actuals)
    - coverage: percentage of predictions that were NOT impossible
      (Surprise sets details['was_impossible']=True when it cannot predict)

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        Dictionary with additional metrics
    """
    if not predictions:
        return {"mse": None, "mape": None, "coverage": None, "n_predictions": 0}

    actuals = np.array([pred.r_ui for pred in predictions], dtype=float)
    estimated = np.array([pred.est for pred in predictions], dtype=float)

    # MSE
    mse = float(np.mean((actuals - estimated) ** 2))

    # MAPE (guard against division by zero)
    non_zero_mask = actuals != 0
    if np.any(non_zero_mask):
        mape = float(
            np.mean(
                np.abs(
                    (actuals[non_zero_mask] - estimated[non_zero_mask])
                    / actuals[non_zero_mask]
                )
            ) * 100
        )
    else:
        mape = None

    # Coverage: share of user-item pairs the model could actually predict
    n_impossible = sum(
        1 for pred in predictions if pred.details.get("was_impossible", False)
    )
    coverage = float((len(predictions) - n_impossible) / len(predictions) * 100)

    return {
        "mse": mse,
        "mape": mape,
        "coverage": coverage,
        "n_predictions": len(predictions),
    }


# =============================================================================
# Visualization Functions (PROVIDED)
# =============================================================================
def create_prediction_distribution_plot(predictions: List) -> plt.Figure:
    """
    Create a plot showing prediction vs actual rating distribution.

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        Matplotlib figure
    """
    actuals = [pred.r_ui for pred in predictions]
    estimated = [pred.est for pred in predictions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Scatter plot of actual vs predicted
    axes[0].scatter(actuals, estimated, alpha=0.1, s=1)
    axes[0].plot([1, 5], [1, 5], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Actual Rating')
    axes[0].set_ylabel('Predicted Rating')
    axes[0].set_title('Actual vs Predicted Ratings')
    axes[0].legend()

    # Plot 2: Distribution of actual ratings
    axes[1].hist(actuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Actual Ratings')

    # Plot 3: Distribution of prediction errors
    errors = np.array(estimated) - np.array(actuals)
    axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--')
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Prediction Errors')

    plt.tight_layout()
    return fig


def create_error_by_rating_plot(predictions: List) -> plt.Figure:
    """
    Create a plot showing error distribution by actual rating.

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        Matplotlib figure
    """
    # Group predictions by actual rating
    rating_groups = {}
    for pred in predictions:
        rating = round(pred.r_ui)
        if rating not in rating_groups:
            rating_groups[rating] = []
        rating_groups[rating].append(pred.est - pred.r_ui)

    fig, ax = plt.subplots(figsize=(10, 6))

    ratings = sorted(rating_groups.keys())
    positions = range(len(ratings))

    bp = ax.boxplot(
        [rating_groups[r] for r in ratings],
        positions=positions,
        widths=0.6
    )

    ax.set_xticklabels([str(r) for r in ratings])
    ax.set_xlabel('Actual Rating')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error by Actual Rating')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    return fig


def save_evaluation_report(metrics: Dict, filepath: str) -> None:
    """
    Save evaluation metrics to a text file.

    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the report
    """
    with open(filepath, 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 40 + "\n\n")

        for name, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{name}: {value:.4f}\n")
            else:
                f.write(f"{name}: {value}\n")

    logger.info(f"Evaluation report saved to {filepath}")


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    print("Testing Evaluation Module")
    print("=" * 50)

    from pipeline.data_ingestion import load_and_split
    from pipeline.training import train_model, setup_mlflow

    setup_mlflow()
    trainset, testset, _ = load_and_split()
    model, run_id = train_model(trainset, model_type="svd", n_factors=50, n_epochs=10)
    metrics = evaluate_model(model, testset, run_id)
    print(f"Metrics: {metrics}")
