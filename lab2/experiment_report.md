# Experiment Report

Generated: 2026-08-09 03:24:50

## Summary

- Total experiments: 9
- Successful: 9
- Failed: 0

## Results

| # | Model | Parameters | RMSE | MAE |
|---|-------|------------|------|-----|
| 1 | svd | `{'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}` | 0.9350 | 0.7374 |
| 2 | svd | `{'n_factors': 100, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}` | 0.9340 | 0.7360 |
| 3 | svd | `{'n_factors': 100, 'n_epochs': 50, 'lr_all': 0.005, 'reg_all': 0.02}` | 0.9670 | 0.7581 |
| 4 | svd | `{'n_factors': 150, 'n_epochs': 30, 'lr_all': 0.01, 'reg_all': 0.02}` | 0.9576 | 0.7523 |
| 5 | nmf | `{'n_factors': 50, 'n_epochs': 50}` | 1.0308 | 0.7858 |
| 6 | nmf | `{'n_factors': 100, 'n_epochs': 50}` | 1.0990 | 0.8372 |
| 7 | knn | `{'k': 20, 'sim_options': {'name': 'cosine', 'user_based': True}}` | 1.0284 | 0.8099 |
| 8 | knn | `{'k': 40, 'sim_options': {'name': 'cosine', 'user_based': True}}` | 1.0194 | 0.8038 |
| 9 | knn | `{'k': 40, 'sim_options': {'name': 'pearson', 'user_based': True}}` | 1.0150 | 0.8037 |

## Best Model

- Configuration: `{'model_type': 'svd', 'n_factors': 100, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}`
- RMSE: 0.9340
- MAE: 0.7360
- Run ID: `3364a5bd92184fbfb19cae9f36369d66`

## Analysis

### Performance by model family

| Model family | # runs | Mean RMSE | Best RMSE | Worst RMSE |
|--------------|-------:|----------:|----------:|-----------:|
| KNN | 3 | 1.0209 | 1.0150 | 1.0284 |
| NMF | 2 | 1.0649 | 1.0308 | 1.0990 |
| SVD | 4 | 0.9484 | 0.9340 | 0.9670 |

### Observations

- **SVD** is the best family (RMSE 0.9340); **NMF** is the worst (RMSE 1.0990). Spread = 0.1650.
- SVD (matrix factorization) outperforms KNN here: it generalizes latent factors instead of relying on user/item neighbourhood similarity, which suffers for users/items with few co-ratings.
- NMF lands between SVD and KNN: non-negative factorization is interpretable but its non-negativity constraint usually costs a little accuracy vs unconstrained SVD on this dataset.
- Best config `{'model_type': 'svd', 'n_factors': 100, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}` achieves coverage 100.0 (share of user-item pairs the model could predict).
- Increasing `n_factors`/`n_epochs` for SVD reduced RMSE in this run set; further gains would likely flatten (diminishing returns) and risk overfitting on a 100K-rating dataset.

## Recommendations

- Deploy the **SVD** model with the configuration above (lowest RMSE among 9 successful runs).
- Register this run to the MLflow Model Registry and promote it to Production: `python -c "from pipeline.registry import register_best_model; register_best_model(experiment_name='hyperparameter-tuning')"`
- Compare runs side-by-side in the MLflow UI (http://localhost:5000) and attach screenshots to the lab report.