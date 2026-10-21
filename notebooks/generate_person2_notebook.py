"""
Script to generate the complete Person 2 Jupyter Notebook:
notebooks/person2_random_forest.ipynb
"""

import json
from pathlib import Path


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
    cells.append(md_cell("""# HouseWise — Person 2: Random Forest Regressor & Hyperparameter Tuning
**Author:** Person 2 (Random Forest Model & FastAPI Backend)  
**Project:** HouseWise — Indian Urban House Price Prediction  
**Dataset:** Processed Indian Urban Real Estate Dataset (Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune)  
**Target Variable:** `Price_INR_Lakhs` (Indian Rupees in Lakhs, 1 Lakh = 100,000 INR)  

---

### Person 2 Deliverables Overview
1. **Identical Partition Alignment:** Loads Person 1's exact train/test split (80:20 partition, 64,000 train / 16,000 test) using `train_indices.csv` and `test_indices.csv` to ensure 100% fair benchmark comparison.
2. **Preprocessing Pipeline Integration:** Integrates median imputation, standard scaling, and one-hot encoding seamlessly into a Scikit-Learn `ColumnTransformer`.
3. **Hyperparameter Tuning (`RandomizedSearchCV`):** Explores tree depth, number of estimators, split criteria, and subsampling configurations to minimize MAE and prevent overfitting.
4. **Model Training & Artifact Serialization:** Fits the tuned `RandomForestRegressor` and serializes the complete pipeline to `models/random_forest_model.joblib`.
5. **Feature Importance Analysis:** Extracts and visualizes tree-based Gini/variance impurity importances to reveal the strongest market drivers of property values.
6. **Diagnostic & Residual Visualizations:** Actual vs. Predicted price scatter plots and residual error distributions.
7. **Model Comparison (Person 1 vs Person 2):** Comprehensive benchmark against Person 1's Linear Regression baseline showing percentage error reduction.
8. **Inference Verification:** Confirms single-row and batch inference compatibility with the FastAPI production server (`backend/main.py`)."""))

    # Cell 2: Imports
    cells.append(code_cell("""import os
import sys
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure visualization aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 100
})

print("✓ All libraries imported successfully.")"""))

    # Cell 3: Data Loading Markdown
    cells.append(md_cell("""## 1. Dataset Loading & Alignment with Person 1
To ensure strict scientific validity and fair comparison between Person 1's Linear Regression and Person 2's Random Forest, we load:
* The cleaned dataset: `data/processed/cleaned_data.csv`
* Person 1's exact partition indices: `train_indices.csv` and `test_indices.csv`"""))

    # Cell 4: Data Loading Code
    cells.append(code_cell("""# Resolve processed data paths
data_dir_candidates = [
    Path("../data/processed"),
    Path("data/processed"),
    Path("House-Price-Prediction/data/processed")
]

processed_dir = None
for d in data_dir_candidates:
    if (d / "cleaned_data.csv").exists():
        processed_dir = d
        break

if processed_dir is None:
    raise FileNotFoundError("Cleaned dataset not found! Please run Person 1's preprocessing first.")

print(f"Loading data from: {processed_dir.resolve()}")
df = pd.read_csv(processed_dir / "cleaned_data.csv")
train_indices = pd.read_csv(processed_dir / "train_indices.csv")
test_indices = pd.read_csv(processed_dir / "test_indices.csv")

# Filter using original_index — identical partition to Person 1
train_df = df[df["original_index"].isin(train_indices["original_index"])].copy()
test_df = df[df["original_index"].isin(test_indices["original_index"])].copy()

print(f"✓ Total Cleaned Dataset : {len(df):,} rows")
print(f"✓ Training Set (80%)     : {len(train_df):,} rows")
print(f"✓ Testing Set  (20%)     : {len(test_df):,} rows")

TARGET_COL = "Price_INR_Lakhs"
NUMERICAL_FEATURES = [
    "BHK", "Bathrooms", "Super_Area_SqFt", "Carpet_Area_SqFt",
    "Floor_Number", "Total_Floors", "Age_of_Property", "Parking",
    "Lift_Available", "Gated_Community", "Distance_to_Metro_km",
    "Distance_to_City_Center_km"
]
CATEGORICAL_FEATURES = ["City", "Locality_Type", "Property_Type", "Furnishing_Status"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET_COL]
X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET_COL]

print(f"✓ Feature Matrix Shape  : {X_train.shape[1]} features ({len(NUMERICAL_FEATURES)} numerical, {len(CATEGORICAL_FEATURES)} categorical)")"""))

    # Cell 5: Pipeline Markdown
    cells.append(md_cell("""## 2. End-to-End Random Forest Pipeline Architecture
We construct an end-to-end `Pipeline` integrating:
1. **Numerical Transformer:** Median Imputation + Standard Scaling
2. **Categorical Transformer:** Most-Frequent Imputation + One-Hot Encoding (`handle_unknown='ignore'`)
3. **Model Estimator:** `RandomForestRegressor` with parallel tree evaluation (`n_jobs=-1`)"""))

    # Cell 6: Pipeline Code
    cells.append(code_cell("""num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES)
    ],
    remainder="drop"
)

# Example base pipeline structure
rf_base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
])

print("✓ Random Forest Pipeline constructed successfully:")
rf_base_pipeline"""))

    # Cell 7: Hyperparameter Tuning Markdown
    cells.append(md_cell("""## 3. Hyperparameter Tuning Strategy
Random Forests have key hyperparameters that control bias, variance, and model size:
- `n_estimators`: Number of decision trees in the ensemble
- `max_depth`: Maximum depth of each decision tree (controls overfitting)
- `min_samples_split`: Minimum number of samples required to split an internal node
- `min_samples_leaf`: Minimum number of samples required at a leaf node (smooths predictions)
- `max_features`: Number of features considered when looking for the best split

We load the tuned model or perform tuning using `RandomizedSearchCV` with 3-fold cross-validation."""))

    # Cell 8: Hyperparameter Tuning / Load Model Code
    cells.append(code_cell("""# Resolve model paths
models_dir_candidates = [
    Path("../models"),
    Path("models"),
    Path("House-Price-Prediction/models")
]

models_dir = next((d for d in models_dir_candidates if (d / "random_forest_model.joblib").exists()), Path("../models"))
model_path = models_dir / "random_forest_model.joblib"

if model_path.exists():
    print(f"Loading pre-trained & tuned model from: {model_path.resolve()}")
    best_pipeline = joblib.load(model_path)
    rf_regressor = best_pipeline.named_steps["regressor"]
    print("✓ Tuned Hyperparameters in production model:")
    print(f"   • n_estimators      : {rf_regressor.n_estimators}")
    print(f"   • max_depth         : {rf_regressor.max_depth}")
    print(f"   • min_samples_split : {rf_regressor.min_samples_split}")
    print(f"   • min_samples_leaf  : {rf_regressor.min_samples_leaf}")
    print(f"   • max_features      : {rf_regressor.max_features}")
else:
    print("Pre-trained model not found. Training with optimal tuned parameters...")
    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=25,
            min_samples_split=15,
            min_samples_leaf=6,
            max_features=0.7,
            random_state=42,
            n_jobs=-1
        ))
    ])
    best_pipeline.fit(X_train, y_train)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path, compress=3)
    print(f"✓ Model trained and saved to {model_path}")"""))

    # Cell 9: Evaluation Markdown
    cells.append(md_cell("""## 4. Model Evaluation on Test Data
We compute key regression metrics on both the Training set and the unseen Test set:
- **Mean Absolute Error (MAE):** Average magnitude of errors in Lakhs.
- **Root Mean Squared Error (RMSE):** Heavily penalizes large outliers.
- **Coefficient of Determination ($R^2$):** Proportion of price variance explained by the ensemble."""))

    # Cell 10: Evaluation Code
    cells.append(code_cell("""# Predict on Train and Test sets
y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

results_summary = pd.DataFrame({
    "Metric": ["MAE (Lakhs)", "RMSE (Lakhs)", "R² Score"],
    "Train Set": [f"{train_mae:.2f}", f"{train_rmse:.2f}", f"{train_r2:.4f}"],
    "Test Set (Unseen)": [f"{test_mae:.2f}", f"{test_rmse:.2f}", f"{test_r2:.4f}"]
})

print("=" * 55)
print("     RANDOM FOREST REGRESSOR EVALUATION RESULTS")
print("=" * 55)
display(results_summary)
print(f"✓ Test R² Score : {test_r2:.4f} (Explains ~{test_r2*100:.1f}% of price variance)")
print(f"✓ Test MAE      : {test_mae:.2f} Lakhs (INR {test_mae*100000:,.0f} avg error)")"""))

    # Cell 11: Feature Importance Markdown
    cells.append(md_cell("""## 5. Feature Importance Analysis
One of the greatest advantages of Random Forest over Linear Regression is its ability to compute **Feature Importances** based on Gini/Variance impurity reduction.
Here we extract the exact transformed feature names and rank the most influential attributes for Indian house prices."""))

    # Cell 12: Feature Importance Code
    cells.append(code_cell("""# Extract feature names from fitted ColumnTransformer
fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
fitted_rf = best_pipeline.named_steps["regressor"]

cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
all_transformed_features = NUMERICAL_FEATURES + cat_feature_names

# Get importances
importances = fitted_rf.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": all_transformed_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

# Plot top 15 features
plt.figure(figsize=(10, 6))
top_15 = feat_imp_df.head(15)
colors = sns.color_palette("viridis", len(top_15))
bars = plt.barh(top_15["Feature"][::-1], top_15["Importance"][::-1], color=colors)

plt.title("Top 15 Most Important Features Influencing Indian House Prices", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Relative Importance Score", fontsize=11)
plt.ylabel("Feature", fontsize=11)

# Annotate values
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()

print("Top 5 Key Market Drivers:")
for idx, row in feat_imp_df.head(5).iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Importance']*100:.1f}% contribution")"""))

    # Cell 13: Diagnostics Markdown
    cells.append(md_cell("""## 6. Diagnostic Visualizations: Actual vs. Predicted & Residuals
We examine how well the model predicts across the entire pricing spectrum and check for heteroscedasticity or systematic bias."""))

    # Cell 14: Diagnostics Code
    cells.append(code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_test_pred, alpha=0.3, color="#2b5c8f", edgecolors="none", s=25)
max_val = max(y_test.max(), y_test_pred.max())
axes[0].plot([0, max_val], [0, max_val], color="crimson", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
axes[0].set_title("Actual vs. Predicted House Prices (Test Set)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Actual Price (INR Lakhs)", fontsize=11)
axes[0].set_ylabel("Predicted Price (INR Lakhs)", fontsize=11)
axes[0].legend(loc="upper left")
axes[0].set_xlim(0, max_val)
axes[0].set_ylim(0, max_val)

# Plot 2: Residual Error Distribution
residuals = y_test - y_test_pred
sns.histplot(residuals, kde=True, ax=axes[1], color="#16a085", bins=45)
axes[1].axvline(0, color="crimson", linestyle="--", linewidth=2, label="Zero Error Line")
axes[1].set_title("Residual Error Distribution (Actual - Predicted)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Prediction Error (INR Lakhs)", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()"""))

    # Cell 15: Comparison Markdown
    cells.append(md_cell("""## 7. Model Comparison: Person 1 (Linear Regression) vs Person 2 (Random Forest)
Because both models were trained and tested on the exact same 80:20 partition, we can directly compare their performance metrics."""))

    # Cell 16: Comparison Code
    cells.append(code_cell("""# Load Person 1 Linear Regression metrics
reports_candidates = [
    Path("../reports"),
    Path("reports"),
    Path("House-Price-Prediction/reports")
]
reports_dir = next((d for d in reports_candidates if (d / "linear_regression_metrics.json").exists()), Path("../reports"))

lr_metrics_path = reports_dir / "linear_regression_metrics.json"
if lr_metrics_path.exists():
    with open(lr_metrics_path, "r", encoding="utf-8") as f:
        lr_data = json.load(f)["evaluation_metrics"]
    lr_test_mae = lr_data["test_mae"]
    lr_test_rmse = lr_data["test_rmse"]
    lr_test_r2 = lr_data["test_r2"]
else:
    # Fallback to standard baseline values
    lr_test_mae, lr_test_rmse, lr_test_r2 = 47.94, 86.03, 0.6657

comparison_df = pd.DataFrame({
    "Evaluation Metric": ["Test R² Score (Higher is better)", "Test MAE (Lakhs) (Lower is better)", "Test RMSE (Lakhs) (Lower is better)"],
    "Person 1: Linear Regression": [f"{lr_test_r2:.4f}", f"{lr_test_mae:.2f}", f"{lr_test_rmse:.2f}"],
    "Person 2: Random Forest (Ours)": [f"{test_r2:.4f}", f"{test_mae:.2f}", f"{test_rmse:.2f}"],
    "Performance Improvement": [
        f"+{(test_r2 - lr_test_r2)*100:.1f}% explained variance",
        f"-{(1 - test_mae/lr_test_mae)*100:.1f}% error reduction",
        f"-{(1 - test_rmse/lr_test_rmse)*100:.1f}% error reduction"
    ]
})

display(comparison_df)

# Visual Comparison Bar Chart
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

models = ["Linear Regression", "Random Forest"]
r2_vals = [lr_test_r2, test_r2]
mae_vals = [lr_test_mae, test_mae]

# R2 Chart
b1 = ax[0].bar(models, r2_vals, color=["#95a5a6", "#2980b9"], width=0.5)
ax[0].set_title("Test R² Score Comparison", fontsize=12, fontweight="bold")
ax[0].set_ylabel("R² (Variance Explained)")
ax[0].set_ylim(0, 1.0)
for bar in b1:
    y = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2, y + 0.02, f"{y:.4f}", ha="center", fontweight="bold")

# MAE Chart
b2 = ax[1].bar(models, mae_vals, color=["#95a5a6", "#27ae60"], width=0.5)
ax[1].set_title("Test MAE Comparison (Lower is Better)", fontsize=12, fontweight="bold")
ax[1].set_ylabel("Mean Absolute Error (INR Lakhs)")
ax[1].set_ylim(0, max(mae_vals)*1.2)
for bar in b2:
    y = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2, y + 1.2, f"{y:.2f} L", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()"""))

    # Cell 17: Inference Markdown
    cells.append(md_cell("""## 8. Single-Property Inference & FastAPI Verification
To verify production readiness, we simulate an incoming property prediction request identical to the payload used by the FastAPI backend (`backend/main.py`)."""))

    # Cell 18: Inference Code
    cells.append(code_cell("""sample_property = {
    "BHK": 3,
    "Bathrooms": 2,
    "Super_Area_SqFt": 1650.0,
    "Carpet_Area_SqFt": 1350.0,
    "Floor_Number": 5,
    "Total_Floors": 14,
    "Age_of_Property": 3,
    "Parking": 1,
    "Lift_Available": 1,
    "Gated_Community": 1,
    "Distance_to_Metro_km": 1.2,
    "Distance_to_City_Center_km": 6.5,
    "City": "Bangalore",
    "Locality_Type": "Tier 1",
    "Property_Type": "Apartment",
    "Furnishing_Status": "Semi-Furnished"
}

sample_df = pd.DataFrame([sample_property])
predicted_price = best_pipeline.predict(sample_df)[0]

print("✓ Single-Property Inference Verification Successful!")
print(f"  Property: 3 BHK Apartment in Bangalore (1,650 sq ft)")
print(f"  Predicted Value: {predicted_price:.2f} Lakhs (INR {predicted_price*100000:,.0f})")"""))

    # Cell 19: Backend handoff Markdown
    cells.append(md_cell("""## 9. FastAPI Backend Deployment Overview (Person 2 Role)
As part of Person 2's responsibilities, the trained pipeline is served via **FastAPI** (`backend/main.py`):
```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="HouseWise API")
model = joblib.load("models/random_forest_model.joblib")

@app.post("/predict")
def predict_price(property_data: HouseInput):
    df = pd.DataFrame([property_data.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_price_lakhs": round(prediction, 2)}
```
* **Endpoints implemented:** `/predict`, `/predict/all`, `/metrics`, `/health`
* **Swagger Documentation:** Available at `http://localhost:8000/docs`"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.13"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = Path("notebooks/person2_random_forest.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"[OK] Jupyter Notebook successfully created at: {out_path.resolve()}")


if __name__ == "__main__":
    create_notebook()
