# HouseWise: 4-Model Comparative Benchmark Report

**Champion Model:** XGBoost Regressor (R² = 0.7896, MAE = 32.49 Lakhs)

| Model | Test MAE (Lakhs) | Test RMSE (Lakhs) | Test R² | MAE Reduction vs Baseline | Status |
|---|---|---|---|---|---|
| **XGBoost Regressor** | 32.4866 | 68.2581 | 0.7896 | 32.24% | Measured |
| **Random Forest Regressor** | 32.9443 | 68.9376 | 0.7853 | 31.28% | Measured |
| **Decision Tree Regressor** | 34.9493 | 72.1917 | 0.7646 | 27.1% | Measured |
| **Linear Regression** | 47.9403 | 86.0255 | 0.6657 | 0.0% | Measured |

### Key Takeaways
1. **Non-linear Capture:** Tree-based models (Decision Tree, Random Forest, XGBoost) significantly outperform Linear Regression.
2. **Gradient Boosting Supremacy:** XGBoost minimizes both bias and variance via sequential tree boosting.
3. **Fair Evaluation:** All models were trained on the exact same 64,000 train samples and evaluated on the same 16,000 test partition.
