"""
HouseWise: XGBoost Regressor Training Module
Author: Person 3 (Decision Tree + XGBoost + Frontend)
Project: HouseWise - Indian Urban House Price Prediction

Trains an XGBoost Regressor using the EXACT same cleaned data
and train/test split as Person 1 and Person 2 for fair, unbiased benchmarking.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# ── Logging Configuration ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_xgboost")

# ── Constants & Paths ──────────────────────────────────────────────────────
TARGET_COL = "Price_INR_Lakhs"
ID_COL = "Property_ID"

DATA_DIR = Path("data/processed")
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
TRAIN_INDICES_PATH = DATA_DIR / "train_indices.csv"
TEST_INDICES_PATH = DATA_DIR / "test_indices.csv"

MODEL_OUTPUT_PATH = Path("models/xgboost_model.joblib")
METRICS_OUTPUT_PATH = Path("reports/xgboost_metrics.json")

# Features matching Person 1 & 2 exactly
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


# ── Data Loading ───────────────────────────────────────────────────────────
def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads cleaned data and splits using Person 1's exact train/test indices.
    """
    logger.info("Loading cleaned data from %s", CLEANED_DATA_PATH)
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

    df = pd.read_csv(CLEANED_DATA_PATH)
    logger.info("Loaded cleaned dataset: %d rows x %d columns", *df.shape)

    train_indices = pd.read_csv(TRAIN_INDICES_PATH)
    test_indices = pd.read_csv(TEST_INDICES_PATH)

    train_df = df[df["original_index"].isin(train_indices["original_index"])].copy()
    test_df = df[df["original_index"].isin(test_indices["original_index"])].copy()

    logger.info("Train set: %d rows | Test set: %d rows", len(train_df), len(test_df))
    return train_df, test_df


# ── Pipeline Construction ──────────────────────────────────────────────────
def build_xgboost_pipeline(
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Constructs an end-to-end Pipeline with preprocessing and XGBRegressor.
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
                XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )

    return model_pipeline


# ── Hyperparameter Tuning ──────────────────────────────────────────────────
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 3,
    n_iter: int = 15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tunes XGBoost gradient boosting hyperparameters via RandomizedSearchCV.
    """
    logger.info(
        "Tuning XGBoost hyperparameters with %d-fold CV, %d iterations...",
        cv_folds,
        n_iter,
    )

    base_pipeline = build_xgboost_pipeline()

    param_distributions = {
        "regressor__n_estimators": [150, 250, 350, 500],
        "regressor__max_depth": [5, 6, 8, 10, 12],
        "regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "regressor__subsample": [0.75, 0.85, 0.95],
        "regressor__colsample_bytree": [0.7, 0.85, 1.0],
        "regressor__reg_alpha": [0.0, 0.1, 1.0],
        "regressor__reg_lambda": [1.0, 2.0, 5.0],
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

    best_params_raw = search.best_params_
    best_params_clean = {
        k.replace("regressor__", ""): v for k, v in best_params_raw.items()
    }
    best_cv_score = -search.best_score_

    logger.info("Best CV MAE: %.4f", best_cv_score)
    logger.info("Best parameters: %s", best_params_clean)

    return {
        "best_params": best_params_clean,
        "best_cv_mae": round(best_cv_score, 4),
        "best_pipeline": search.best_estimator_,
    }


# ── Model Evaluation ───────────────────────────────────────────────────────
def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluates XGBoost on train and test sets (MAE, RMSE, R²).
    """
    y_train_pred = pipeline.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    train_r2 = r2_score(y_train, y_train_pred)

    y_test_pred = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_r2 = r2_score(y_test, y_test_pred)

    metrics = {
        "model_name": "XGBoost Regressor (Tuned Pipeline)",
        "project": "HouseWise",
        "target_variable": TARGET_COL,
        "target_unit": "INR Lakhs",
        "partitions": {
            "training_samples": len(y_train),
            "testing_samples": len(y_test),
            "split_ratio": "80:20 (random_state=42)",
        },
        "evaluation_metrics": {
            "test_mae": round(float(test_mae), 4),
            "test_rmse": round(float(test_rmse), 4),
            "test_r2": round(float(test_r2), 4),
            "train_mae": round(float(train_mae), 4),
            "train_rmse": round(float(train_rmse), 4),
            "train_r2": round(float(train_r2), 4),
        },
        "interpretation_guide": {
            "mae_interpretation": (
                "Mean Absolute Error: On average, the XGBoost prediction "
                "deviates by this amount in Lakhs from actual house price. Lower is better."
            ),
            "rmse_interpretation": (
                "Root Mean Squared Error: Penalizes larger pricing errors more heavily. Lower is better."
            ),
            "r2_interpretation": (
                "R-squared coefficient of determination: Proportion of price variance "
                "explained by sequentially boosted regression trees. Higher is better (1.0 = perfect)."
            ),
        },
    }

    logger.info("=" * 60)
    logger.info("XGBOOST EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("TRAIN — MAE: %.4f | RMSE: %.4f | R²: %.4f", train_mae, train_rmse, train_r2)
    logger.info("TEST  — MAE: %.4f | RMSE: %.4f | R²: %.4f", test_mae, test_rmse, test_r2)
    logger.info("=" * 60)

    return metrics


# ── Training Orchestration ──────────────────────────────────────────────────
def run_training(tune: bool = True, cv_folds: int = 3, n_iter: int = 15) -> Dict[str, Any]:
    """
    Executes end-to-end training, evaluation, and serialization.
    """
    train_df, test_df = load_train_test_data()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COL]

    if tune:
        tuning_results = tune_hyperparameters(
            X_train, y_train, cv_folds=cv_folds, n_iter=n_iter
        )
        best_pipeline = tuning_results["best_pipeline"]
        best_params = tuning_results["best_params"]
    else:
        best_pipeline = build_xgboost_pipeline()
        best_pipeline.fit(X_train, y_train)
        best_params = {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        }

    metrics = evaluate_model(best_pipeline, X_train, y_train, X_test, y_test)
    metrics["hyperparameters"] = best_params

    # Save model artifact
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    logger.info("Saved XGBoost pipeline to %s", MODEL_OUTPUT_PATH)

    # Save metrics JSON
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_OUTPUT_PATH)

    return metrics


if __name__ == "__main__":
    run_training(tune=True, cv_folds=3, n_iter=15)
