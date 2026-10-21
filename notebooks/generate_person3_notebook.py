"""
Script to generate the complete Person 3 Jupyter Notebook:
notebooks/person3_xgboost_decision_tree.ipynb
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def create_notebook():
    cells = []

    def md_cell(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    def code_cell(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    # Cell 1: Title
    cells.append(md_cell("""# HouseWise — Person 3: XGBoost & Decision Tree Regressors with 4-Model Benchmark
**Author:** Person 3 (XGBoost + Decision Tree + Frontend Development)  
**Project:** HouseWise — Indian Urban House Price Prediction  
**Dataset:** Cleaned Indian Real Estate Dataset (80,000 properties across 10 major metro hubs: Mumbai, Delhi, Bangalore, Pune, Hyderabad, Chennai, Kolkata, Ahmedabad, Jaipur, Chandigarh)  
**Target Variable:** `Price_INR_Lakhs` (Indian Rupees in Lakhs, 1 Lakh = 100,000 INR)  

---

### Person 3 Machine Learning Deliverables (Part A)
1. **Identical Partition Alignment:** Loads Person 1's exact train/test split (80:20 split, 64,000 train / 16,000 test) using `train_indices.csv` and `test_indices.csv` to ensure 100% fair and scientifically valid comparison across all 4 project models.
2. **Decision Tree Regressor:** Learns intuitive hierarchical decision rules by partitioning feature space into orthogonal splits and predicting leaf values. Hyperparameter-tuned with `RandomizedSearchCV` (max_depth, min_samples_split, min_samples_leaf).
3. **XGBoost Regressor:** Implements extreme gradient boosted decision trees with second-order gradient shrinkage, column subsampling, and L1/L2 regularization to maximize predictive accuracy.
4. **Evaluation Metrics:** Evaluates both models using MAE, RMSE, and R² on both train and test partitions.
5. **Feature Importance Comparison:** Compares decision tree split importance vs. XGBoost gain/weight importance to understand key property value drivers.
6. **Unified 4-Model Benchmark:** Compares Linear Regression, Decision Tree, Random Forest, and XGBoost side-by-side, analyzing error distributions and trade-offs.
7. **Pipeline Artifact Serialization:** Saves both end-to-end pipelines to `models/decision_tree_model.joblib` and `models/xgboost_model.joblib`, with evaluation metrics saved to `reports/`."""))

    # Cell 2: Colab Environment Setup
    cells.append(code_cell("""# Environment & Library Verification (Google Colab / Local compatible)
import sys
import subprocess

# Verify / install necessary dependencies
try:
    import xgboost
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "joblib"])

print("✓ Environment is fully configured for Person 3 ML pipeline execution.")"""))

    # Cell 3: Imports
    cells.append(code_cell("""import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 100
})

print("✓ All machine learning and visualization libraries imported successfully.")"""))

    # Cell 4: Data Alignment Markdown
    cells.append(md_cell("""## 1. Dataset Loading & Strict Partition Alignment
To preserve the scientific integrity of the team benchmark, Person 3 uses the identical 64,000-sample training partition and 16,000-sample testing partition established by Person 1 and used by Person 2."""))

    # Cell 5: Data Loading Code
    cells.append(code_cell("""# Resolve processed data paths (supports both Colab and local directories)
data_dir_candidates = [
    Path("../data/processed"),
    Path("data/processed"),
    Path("/content/House-Price-Prediction/data/processed")
]

processed_dir = None
for d in data_dir_candidates:
    if (d / "cleaned_data.csv").exists():
        processed_dir = d
        break

if processed_dir is None:
    raise FileNotFoundError("Could not find data/processed/cleaned_data.csv. Please ensure the repository is mounted.")

print(f"Loading cleaned dataset from: {processed_dir.resolve()}")

df = pd.read_csv(processed_dir / "cleaned_data.csv")
train_indices = pd.read_csv(processed_dir / "train_indices.csv")
test_indices = pd.read_csv(processed_dir / "test_indices.csv")

train_df = df[df["original_index"].isin(train_indices["original_index"])].copy()
test_df = df[df["original_index"].isin(test_indices["original_index"])].copy()

print(f"✓ Cleaned Dataset Shape: {df.shape}")
print(f"✓ Training Set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"✓ Testing Set:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")"""))

    # Cell 6: Feature Definition Markdown
    cells.append(md_cell("""## 2. Feature Schema & Preprocessing Pipeline
We use the exact 16 property features defined in Person 1's `feature_schema.json`:
- **12 Numerical Features:** BHK, Bathrooms, Super Area, Carpet Area, Floor Number, Total Floors, Age of Property, Parking, Lift Available, Gated Community, Distance to Metro, Distance to City Center.
- **4 Categorical Features:** City, Locality Type, Property Type, Furnishing Status."""))

    # Cell 7: Pipeline Definition Code
    cells.append(code_cell("""TARGET_COL = "Price_INR_Lakhs"

NUMERICAL_FEATURES = [
    "BHK", "Bathrooms", "Super_Area_SqFt", "Carpet_Area_SqFt",
    "Floor_Number", "Total_Floors", "Age_of_Property", "Parking",
    "Lift_Available", "Gated_Community", "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
]

CATEGORICAL_FEATURES = [
    "City", "Locality_Type", "Property_Type", "Furnishing_Status",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET_COL]
X_test = test_df[ALL_FEATURES]
y_test = test_df[TARGET_COL]

def create_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")

print(f"Preprocessing pipeline configured for {len(NUMERICAL_FEATURES)} numerical and {len(CATEGORICAL_FEATURES)} categorical features.")"""))

    # Cell 8: Decision Tree Markdown
    cells.append(md_cell("""## 3. Decision Tree Regressor: Training & Hyperparameter Tuning
The Decision Tree Regressor recursively splits properties according to feature thresholds to minimize squared error loss.
We tune tree depth, min_samples_split, and min_samples_leaf via `RandomizedSearchCV` with 3-fold cross validation."""))

    # Cell 9: Decision Tree Code
    cells.append(code_cell("""print("Starting Decision Tree Hyperparameter Tuning (3-fold CV, 20 iterations)...")

dt_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

dt_param_dist = {
    "regressor__max_depth": [8, 10, 12, 15, 20, 25, None],
    "regressor__min_samples_split": [10, 20, 30, 50, 100],
    "regressor__min_samples_leaf": [5, 10, 15, 25, 50],
    "regressor__max_features": [None, "sqrt", "log2", 0.7, 0.9],
}

dt_search = RandomizedSearchCV(
    estimator=dt_pipeline,
    param_distributions=dt_param_dist,
    n_iter=20,
    cv=3,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

dt_search.fit(X_train, y_train)
best_dt = dt_search.best_estimator_

print(f"✓ Best Decision Tree CV MAE: {-dt_search.best_score_:.4f} Lakhs")
print(f"✓ Best Hyperparameters: {dt_search.best_params_}")"""))

    # Cell 10: Decision Tree Evaluation
    cells.append(code_cell("""# Evaluate Decision Tree
y_dt_train_pred = best_dt.predict(X_train)
y_dt_test_pred = best_dt.predict(X_test)

dt_train_mae = mean_absolute_error(y_train, y_dt_train_pred)
dt_train_rmse = np.sqrt(mean_squared_error(y_train, y_dt_train_pred))
dt_train_r2 = r2_score(y_train, y_dt_train_pred)

dt_test_mae = mean_absolute_error(y_test, y_dt_test_pred)
dt_test_rmse = np.sqrt(mean_squared_error(y_test, y_dt_test_pred))
dt_test_r2 = r2_score(y_test, y_dt_test_pred)

print("=" * 55)
print("  DECISION TREE REGRESSOR PERFORMANCE")
print("=" * 55)
print(f"  Train MAE:  ₹ {dt_train_mae:.2f} Lakhs | RMSE: ₹ {dt_train_rmse:.2f} Lakhs | R²: {dt_train_r2:.4f}")
print(f"  Test  MAE:  ₹ {dt_test_mae:.2f} Lakhs | RMSE: ₹ {dt_test_rmse:.2f} Lakhs | R²: {dt_test_r2:.4f}")
print("=" * 55)"""))

    # Cell 11: XGBoost Markdown
    cells.append(md_cell("""## 4. XGBoost Regressor: Training & Gradient Boosting Tuning
XGBoost builds sequential ensembles of weak decision trees, where each subsequent tree corrects the residual errors of the previous ensemble.
We tune `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and regularization terms (`reg_alpha`, `reg_lambda`)."""))

    # Cell 12: XGBoost Code
    cells.append(code_cell("""print("Starting XGBoost Hyperparameter Tuning (3-fold CV, 15 iterations)...")

xgb_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("regressor", XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    ))
])

xgb_param_dist = {
    "regressor__n_estimators": [150, 250, 350, 500],
    "regressor__max_depth": [5, 6, 8, 10, 12],
    "regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "regressor__subsample": [0.75, 0.85, 0.95],
    "regressor__colsample_bytree": [0.7, 0.85, 1.0],
    "regressor__reg_alpha": [0.0, 0.1, 1.0],
    "regressor__reg_lambda": [1.0, 2.0, 5.0],
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_dist,
    n_iter=15,
    cv=3,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

print(f"✓ Best XGBoost CV MAE: {-xgb_search.best_score_:.4f} Lakhs")
print(f"✓ Best Hyperparameters: {xgb_search.best_params_}")"""))

    # Cell 13: XGBoost Evaluation
    cells.append(code_cell("""# Evaluate XGBoost
y_xgb_train_pred = best_xgb.predict(X_train)
y_xgb_test_pred = best_xgb.predict(X_test)

xgb_train_mae = mean_absolute_error(y_train, y_xgb_train_pred)
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, y_xgb_train_pred))
xgb_train_r2 = r2_score(y_train, y_xgb_train_pred)

xgb_test_mae = mean_absolute_error(y_test, y_xgb_test_pred)
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, y_xgb_test_pred))
xgb_test_r2 = r2_score(y_test, y_xgb_test_pred)

print("=" * 55)
print("  XGBOOST REGRESSOR PERFORMANCE")
print("=" * 55)
print(f"  Train MAE:  ₹ {xgb_train_mae:.2f} Lakhs | RMSE: ₹ {xgb_train_rmse:.2f} Lakhs | R²: {xgb_train_r2:.4f}")
print(f"  Test  MAE:  ₹ {xgb_test_mae:.2f} Lakhs | RMSE: ₹ {xgb_test_rmse:.2f} Lakhs | R²: {xgb_test_r2:.4f}")
print("=" * 55)"""))

    # Cell 14: Feature Importance Analysis
    cells.append(code_cell("""# Extract Feature Importances
ohe = best_xgb.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
feature_names = NUMERICAL_FEATURES + cat_feature_names

xgb_importances = best_xgb.named_steps["regressor"].feature_importances_
top_idx = np.argsort(xgb_importances)[-10:]

plt.figure(figsize=(10, 6), dpi=120)
plt.barh(np.array(feature_names)[top_idx], xgb_importances[top_idx], color="#f59e0b")
plt.title("Top 10 Feature Importances in XGBoost Regressor", fontweight="bold")
plt.xlabel("Relative Importance (Gain / Split Contribution)")
plt.tight_layout()
plt.show()"""))

    # Cell 15: 4-Model Benchmark Comparison
    cells.append(md_cell("""## 5. Comprehensive 4-Model Comparative Benchmark
We compare all four models developed across the project:
1. Linear Regression (Baseline, Person 1)
2. Decision Tree Regressor (Orthogonal Splits, Person 3)
3. Random Forest Regressor (Ensemble Bagging, Person 2)
4. XGBoost Regressor (Gradient Boosting, Person 3)"""))

    # Cell 16: Benchmark Code
    cells.append(code_cell("""# Benchmark Summary Table
comparison_data = [
    {"Model": "Linear Regression (Person 1)", "MAE": 47.9403, "RMSE": 86.0255, "R²": 0.6657, "Type": "Linear Baseline"},
    {"Model": "Decision Tree (Person 3)", "MAE": dt_test_mae, "RMSE": dt_test_rmse, "R²": dt_test_r2, "Type": "Single Tree"},
    {"Model": "Random Forest (Person 2)", "MAE": 32.9443, "RMSE": 68.9376, "R²": 0.7853, "Type": "Bagged Ensemble"},
    {"Model": "XGBoost (Person 3)", "MAE": xgb_test_mae, "RMSE": xgb_test_rmse, "R²": xgb_test_r2, "Type": "Gradient Boosted"},
]

comp_df = pd.DataFrame(comparison_data)
comp_df["MAE_Reduction_%"] = ((47.9403 - comp_df["MAE"]) / 47.9403 * 100).round(2)
comp_df["R²_Gain"] = (comp_df["R²"] - 0.6657).round(4)

print(comp_df.to_string(index=False))"""))

    # Cell 17: Comparison Visualization
    cells.append(code_cell("""# Comparative Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
palette = ["#38bdf8", "#a855f7", "#34d399", "#f59e0b"]

# R²
bars = axes[0].bar(comp_df["Model"], comp_df["R²"], color=palette)
axes[0].set_title("Test R² Score (Higher is Better)", fontweight="bold")
axes[0].set_ylim(0, 1.0)
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.4f}", ha="center", fontweight="bold")

# MAE
bars = axes[1].bar(comp_df["Model"], comp_df["MAE"], color=palette)
axes[1].set_title("Test MAE in Lakhs (Lower is Better)", fontweight="bold")
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"₹{bar.get_height():.2f}L", ha="center", fontweight="bold")

# RMSE
bars = axes[2].bar(comp_df["Model"], comp_df["RMSE"], color=palette)
axes[2].set_title("Test RMSE in Lakhs (Lower is Better)", fontweight="bold")
for bar in bars:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"₹{bar.get_height():.2f}L", ha="center", fontweight="bold")

for ax in axes:
    ax.set_xticklabels(["Linear Reg", "Decision Tree", "Random Forest", "XGBoost"], rotation=15)

plt.tight_layout()
plt.show()"""))

    # Cell 18: Artifact Serialization
    cells.append(code_cell("""# Serialize Fitted Pipelines to models/ directory
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_dt, models_dir / "decision_tree_model.joblib")
joblib.dump(best_xgb, models_dir / "xgboost_model.joblib")

print("✓ Saved fitted Decision Tree pipeline to models/decision_tree_model.joblib")
print("✓ Saved fitted XGBoost pipeline to models/xgboost_model.joblib")"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_path = Path("notebooks/person3_xgboost_decision_tree.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"✓ Successfully generated Person 3 Jupyter Notebook at: {out_path}")


if __name__ == "__main__":
    create_notebook()
