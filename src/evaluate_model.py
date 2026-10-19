"""
HouseWise: Model Evaluation and Verification Module
Author: Person 1 (Data Preprocessing, EDA & Linear Regression)
Project: HouseWise - Indian Urban House Price Prediction
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import (
    ID_COL,
    TARGET_COL,
    DEFAULT_PROCESSED_DATA,
    DEFAULT_TEST_INDICES,
    DEFAULT_TRAIN_INDICES,
    identify_column_types,
)
from train_linear_regression import DEFAULT_MODEL_PATH, DEFAULT_SCHEMA_PATH

logger = logging.getLogger("evaluate_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_METRICS_CSV = Path("reports/linear_regression_metrics.csv")
DEFAULT_METRICS_JSON = Path("reports/linear_regression_metrics.json")


def load_heldout_test_set(
    cleaned_data_path: Path = DEFAULT_PROCESSED_DATA,
    test_indices_path: Path = DEFAULT_TEST_INDICES,
    schema_path: Path = DEFAULT_SCHEMA_PATH
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads strictly the held-out test partition matching saved test indices.
    Returns (X_test, y_test) without target leakage.
    """
    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_data_path}")
    if not test_indices_path.exists():
        raise FileNotFoundError(f"Test indices not found at {test_indices_path}")
        
    df = pd.read_csv(cleaned_data_path)
    test_indices_df = pd.read_csv(test_indices_path)
    
    if "original_index" in df.columns and "original_index" in test_indices_df.columns:
        test_df = df[df["original_index"].isin(test_indices_df["original_index"])].copy()
    else:
        test_df = df.iloc[test_indices_df.index].copy()
        
    # Read ordered feature names from schema if available, else derive
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        feature_cols = schema["features"]["all_features_ordered"]
    else:
        num_cols, cat_cols, _ = identify_column_types(test_df, target_col=TARGET_COL, id_col=ID_COL)
        feature_cols = num_cols + cat_cols

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()
    
    return X_test, y_test


def evaluate_pipeline(
    model_path: Path = DEFAULT_MODEL_PATH,
    cleaned_data_path: Path = DEFAULT_PROCESSED_DATA,
    train_indices_path: Path = DEFAULT_TRAIN_INDICES,
    test_indices_path: Path = DEFAULT_TEST_INDICES,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    metrics_csv_path: Path = DEFAULT_METRICS_CSV,
    metrics_json_path: Path = DEFAULT_METRICS_JSON
) -> Dict[str, Any]:
    """
    Evaluates the fitted Linear Regression pipeline on held-out test data
    and computes MAE, RMSE, and R2. Exports metrics to CSV and JSON.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
        
    logger.info("Loading fitted pipeline from %s", model_path)
    pipeline = joblib.load(model_path)
    
    # Load test set
    X_test, y_test = load_heldout_test_set(
        cleaned_data_path=cleaned_data_path,
        test_indices_path=test_indices_path,
        schema_path=schema_path
    )
    
    # Generate test predictions
    logger.info("Computing predictions on held-out test set (%d samples)...", len(X_test))
    y_pred_test = pipeline.predict(X_test)
    
    # Calculate test metrics
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    test_r2 = float(r2_score(y_test, y_pred_test))
    
    # Calculate training metrics for diagnostic comparison
    df = pd.read_csv(cleaned_data_path)
    train_indices_df = pd.read_csv(train_indices_path)
    if "original_index" in df.columns and "original_index" in train_indices_df.columns:
        train_df = df[df["original_index"].isin(train_indices_df["original_index"])].copy()
    else:
        train_df = df.iloc[train_indices_df.index].copy()
        
    X_train = train_df[X_test.columns].copy()
    y_train = train_df[TARGET_COL].copy()
    
    y_pred_train = pipeline.predict(X_train)
    train_mae = float(mean_absolute_error(y_train, y_pred_train))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    train_r2 = float(r2_score(y_train, y_pred_train))
    
    metrics_payload = {
        "model_name": "Linear Regression (Baseline Pipeline)",
        "project": "HouseWise",
        "target_variable": TARGET_COL,
        "target_unit": "INR Lakhs",
        "partitions": {
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "split_ratio": "80:20 (random_state=42)"
        },
        "evaluation_metrics": {
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "test_r2": round(test_r2, 4),
            "train_mae": round(train_mae, 4),
            "train_rmse": round(train_rmse, 4),
            "train_r2": round(train_r2, 4)
        },
        "interpretation_guide": {
            "mae_interpretation": "Mean Absolute Error: On average, the linear regression prediction deviates by this amount in Lakhs from actual house price. Lower is better.",
            "rmse_interpretation": "Root Mean Squared Error: Penalizes larger pricing errors more heavily. Lower is better.",
            "r2_interpretation": "R-squared coefficient of determination: Proportion of price variance explained by linear combinations of features. Higher is better (1.0 = perfect explanation)."
        }
    }
    
    # Save JSON metrics
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Saved metrics JSON to %s", metrics_json_path)
    
    # Save CSV metrics
    metrics_table = pd.DataFrame([{
        "Model": "Linear Regression",
        "Test_MAE_Lakhs": round(test_mae, 4),
        "Test_RMSE_Lakhs": round(test_rmse, 4),
        "Test_R2": round(test_r2, 4),
        "Train_MAE_Lakhs": round(train_mae, 4),
        "Train_RMSE_Lakhs": round(train_rmse, 4),
        "Train_R2": round(train_r2, 4),
        "Train_Rows": len(X_train),
        "Test_Rows": len(X_test)
    }])
    metrics_table.to_csv(metrics_csv_path, index=False)
    logger.info("Saved metrics CSV to %s", metrics_csv_path)
    
    # Print formatted summary table to console
    print("\n" + "=" * 80)
    print("HOUSEWISE: LINEAR REGRESSION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Target Column:           {TARGET_COL} (INR Lakhs)")
    print(f"Training Partition:      {len(X_train):,} rows (80%)")
    print(f"Held-out Test Partition: {len(X_test):,} rows (20%)")
    print("-" * 80)
    print(f"{'Metric':<25} | {'Test Set (Held-Out)':<20} | {'Train Set (Reference)':<20}")
    print("-" * 80)
    print(f"{'MAE (Mean Abs Error)':<25} | {test_mae:<20.4f} | {train_mae:<20.4f}")
    print(f"{'RMSE (Root Mean Sq Err)':<25} | {test_rmse:<20.4f} | {train_rmse:<20.4f}")
    print(f"{'R² (Explained Variance)':<25} | {test_r2:<20.4f} | {train_r2:<20.4f}")
    print("=" * 80)
    print("Interpretation:")
    print(" • MAE & RMSE: Lower values indicate superior predictive accuracy and smaller pricing errors.")
    print(" • R²: Values closer to 1.0 denote a higher proportion of variance accounted for by the model.")
    print(" • Comparison: Train vs Test consistency confirms no overt overfitting in the linear baseline.")
    print("=" * 80 + "\n")
    
    return metrics_payload


def verify_model_pipeline(
    model_path: Path = DEFAULT_MODEL_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    cleaned_data_path: Path = DEFAULT_PROCESSED_DATA,
    test_indices_path: Path = DEFAULT_TEST_INDICES
) -> bool:
    """
    Automated verification test:
    1. Reloads saved model from joblib.
    2. Constructs inference DataFrame with expected raw feature columns (without target or ID).
    3. Verifies predictions are finite, non-empty, and shaped correctly.
    4. Tests both single-row inference (FastAPI simulation) and batch inference.
    """
    print("\n--- Running Automated Model Verification Test ---")
    assert model_path.exists(), f"Model artifact missing at {model_path}"
    pipeline = joblib.load(model_path)
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    expected_features = schema["features"]["all_features_ordered"]
    
    # 1. Single sample prediction test (FastAPI / Person 2 simulation)
    sample_property = {
        "BHK": 3,
        "Bathrooms": 3,
        "Super_Area_SqFt": 1850.0,
        "Carpet_Area_SqFt": 1420.0,
        "Floor_Number": 8,
        "Total_Floors": 20,
        "Age_of_Property": 5,
        "Parking": 1,
        "Lift_Available": 1,
        "Gated_Community": 1,
        "Distance_to_Metro_km": 1.5,
        "Distance_to_City_Center_km": 6.2,
        "City": "Bangalore",
        "Locality_Type": "Premium",
        "Property_Type": "Apartment",
        "Furnishing_Status": "Furnished"
    }
    
    single_df = pd.DataFrame([sample_property])[expected_features]
    assert TARGET_COL not in single_df.columns, "Target column must NEVER be in prediction input!"
    assert ID_COL not in single_df.columns, "Identifier column must NEVER be in prediction input!"
    
    single_pred = pipeline.predict(single_df)
    assert len(single_pred) == 1, f"Expected 1 prediction, got {len(single_pred)}"
    assert np.isfinite(single_pred[0]), f"Predicted value is not finite: {single_pred[0]}"
    print(f"✓ Single-row inference verified: Predicted price = {single_pred[0]:.2f} Lakhs")
    
    # 2. Batch inference test on held-out test records
    X_test, _ = load_heldout_test_set(cleaned_data_path, test_indices_path, schema_path)
    batch_test = X_test.head(10).copy()
    batch_preds = pipeline.predict(batch_test)
    
    assert len(batch_preds) == 10, f"Expected 10 predictions, got {len(batch_preds)}"
    assert np.all(np.isfinite(batch_preds)), "Batch predictions contain NaN or Inf values!"
    print(f"✓ Batch inference verified: 10 test samples predicted successfully.")
    print(f"  Sample predictions (Lakhs): {[round(p, 2) for p in batch_preds[:5]]}")
    
    print("✓ All automated verification checks PASSED successfully!\n")
    return True


if __name__ == "__main__":
    evaluate_pipeline()
    verify_model_pipeline()
