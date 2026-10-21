"""
HouseWise: Random Forest Regressor Training Module
Author: Person 2 (Random Forest + FastAPI Backend)
Project: HouseWise - Indian Urban House Price Prediction

Uses the EXACT same cleaned data and train/test split as Person 1
to ensure fair model comparison.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_random_forest")

# ── Constants (same as Person 1) ─────────────────────────────────────────────
TARGET_COL = "Price_INR_Lakhs"
ID_COL = "Property_ID"

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
TRAIN_INDICES_PATH = DATA_DIR / "train_indices.csv"
TEST_INDICES_PATH = DATA_DIR / "test_indices.csv"

MODEL_OUTPUT_PATH = Path("models/random_forest_model.joblib")
METRICS_OUTPUT_PATH = Path("reports/random_forest_metrics.json")

# Feature lists — must match Person 1's feature_schema.json exactly
NUMERICAL_FEATURES = [
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
]

CATEGORICAL_FEATURES = [
    "City",
    "Locality_Type",
    "Property_Type",
    "Furnishing_Status",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads cleaned data and splits it using Person 1's exact train/test indices.
    This ensures all team members train and evaluate on identical partitions.
    """
    logger.info("Loading cleaned data from %s", CLEANED_DATA_PATH)
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEANED_DATA_PATH}. "
            "Ensure Person 1's preprocessing has been run first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    logger.info("Loaded cleaned dataset: %d rows x %d columns", *df.shape)

    # Load train/test indices
    for path, label in [(TRAIN_INDICES_PATH, "train"), (TEST_INDICES_PATH, "test")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label.capitalize()} indices not found at {path}. "
                "Ensure Person 1's preprocessing has been run first."
            )

    train_indices = pd.read_csv(TRAIN_INDICES_PATH)
    test_indices = pd.read_csv(TEST_INDICES_PATH)

    # Filter using original_index — same method as Person 1
    train_df = df[df["original_index"].isin(train_indices["original_index"])].copy()
    test_df = df[df["original_index"].isin(test_indices["original_index"])].copy()

    logger.info("Train set: %d rows | Test set: %d rows", len(train_df), len(test_df))

    return train_df, test_df


# ── Pipeline Construction ────────────────────────────────────────────────────
def build_random_forest_pipeline(
    n_estimators: int = 200,
    max_depth: int = 20,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    random_state: int = 42,
) -> Pipeline:
    """
    Constructs an end-to-end Pipeline:
    Preprocessor (identical to Person 1):
      - Numerical: SimpleImputer(median) -> StandardScaler()
      - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    Estimator:
      - RandomForestRegressor with specified hyperparameters
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model_pipeline


# ── Hyperparameter Tuning ────────────────────────────────────────────────────
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 3,
    n_iter: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Uses RandomizedSearchCV to find the best Random Forest hyperparameters.
    Searches over n_estimators, max_depth, min_samples_split, min_samples_leaf,
    and max_features.
    """
    logger.info(
        "Starting hyperparameter tuning with %d-fold CV, %d iterations...",
        cv_folds,
        n_iter,
    )

    # Build base pipeline (default params, will be overridden by search)
    base_pipeline = build_random_forest_pipeline()

    # Parameter grid — prefixed with 'regressor__' because the estimator
    # is inside the pipeline under the key 'regressor'
    param_distributions = {
        "regressor__n_estimators": [100, 150, 200, 300, 400],
        "regressor__max_depth": [10, 15, 20, 25, 30, None],
        "regressor__min_samples_split": [2, 5, 10, 15],
        "regressor__min_samples_leaf": [1, 2, 4, 6],
        "regressor__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    # Extract best params (strip the 'regressor__' prefix for readability)
    best_params_raw = search.best_params_
    best_params_clean = {
        k.replace("regressor__", ""): v for k, v in best_params_raw.items()
    }

    best_cv_score = -search.best_score_  # negate because scoring is neg_mae

    logger.info("Best CV MAE: %.4f", best_cv_score)
    logger.info("Best parameters: %s", best_params_clean)

    return {
        "best_params": best_params_clean,
        "best_cv_mae": round(best_cv_score, 4),
        "best_pipeline": search.best_estimator_,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluates the trained pipeline on both train and test sets.
    Computes MAE, RMSE, and R² — same metrics as Person 1.
    """
    # Training set predictions
    y_train_pred = pipeline.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Test set predictions
    y_test_pred = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    metrics = {
        "model_name": "Random Forest Regressor (Tuned Pipeline)",
        "project": "HouseWise",
        "target_variable": TARGET_COL,
        "target_unit": "INR Lakhs",
        "partitions": {
            "training_samples": len(y_train),
            "testing_samples": len(y_test),
            "split_ratio": "80:20 (random_state=42)",
        },
        "evaluation_metrics": {
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "test_r2": round(test_r2, 4),
            "train_mae": round(train_mae, 4),
            "train_rmse": round(train_rmse, 4),
            "train_r2": round(train_r2, 4),
        },
        "interpretation_guide": {
            "mae_interpretation": (
                "Mean Absolute Error: On average, the Random Forest prediction "
                "deviates by this amount in Lakhs from actual house price. Lower is better."
            ),
            "rmse_interpretation": (
                "Root Mean Squared Error: Penalizes larger pricing errors more "
                "heavily. Lower is better."
            ),
            "r2_interpretation": (
                "R-squared coefficient of determination: Proportion of price variance "
                "explained by the ensemble of decision trees. Higher is better "
                "(1.0 = perfect explanation)."
            ),
        },
    }

    logger.info("=" * 60)
    logger.info("RANDOM FOREST EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("TRAIN — MAE: %.4f | RMSE: %.4f | R²: %.4f", train_mae, train_rmse, train_r2)
    logger.info("TEST  — MAE: %.4f | RMSE: %.4f | R²: %.4f", test_mae, test_rmse, test_r2)
    logger.info("=" * 60)

    return metrics


# ── Main Orchestration ───────────────────────────────────────────────────────
def run_training(tune: bool = True, cv_folds: int = 3, n_iter: int = 20) -> None:
    """
    Full training pipeline for Random Forest:
    1. Load Person 1's exact train/test split
    2. Optionally tune hyperparameters via RandomizedSearchCV
    3. Evaluate on both train and test sets
    4. Save model pipeline (.joblib) and metrics (.json)
    """
    # 1. Load data
    train_df, test_df = load_train_test_data()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COL]

    logger.info("Feature matrix shapes — Train: %s, Test: %s", X_train.shape, X_test.shape)

    # 2. Train (with optional tuning)
    if tune:
        tuning_results = tune_hyperparameters(
            X_train, y_train, cv_folds=cv_folds, n_iter=n_iter
        )
        best_pipeline = tuning_results["best_pipeline"]
        best_params = tuning_results["best_params"]
        logger.info("Using tuned pipeline with params: %s", best_params)
    else:
        logger.info("Training Random Forest with default hyperparameters...")
        best_pipeline = build_random_forest_pipeline()
        best_pipeline.fit(X_train, y_train)
        best_params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }

    # 3. Evaluate
    metrics = evaluate_model(best_pipeline, X_train, y_train, X_test, y_test)
    metrics["hyperparameters"] = best_params

    # 4. Save model
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    logger.info("Saved fitted Random Forest pipeline to %s", MODEL_OUTPUT_PATH)

    # 5. Save metrics
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", METRICS_OUTPUT_PATH)

    print("\n[OK] Random Forest training complete!")
    print(f"   Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"   Metrics saved to: {METRICS_OUTPUT_PATH}")


if __name__ == "__main__":
    run_training(tune=True, cv_folds=3, n_iter=20)
