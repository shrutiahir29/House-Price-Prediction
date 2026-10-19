"""
HouseWise: Linear Regression Model Training Module
Author: Person 1 (Data Preprocessing, EDA & Linear Regression)
Project: HouseWise - Indian Urban House Price Prediction
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    ID_COL,
    TARGET_COL,
    DEFAULT_PROCESSED_DATA,
    DEFAULT_TRAIN_INDICES,
    identify_column_types,
    load_raw_data,
)

logger = logging.getLogger("train_linear_regression")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_MODEL_PATH = Path("models/linear_regression_model.joblib")
DEFAULT_SCHEMA_PATH = Path("reports/feature_schema.json")


def build_linear_regression_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> Pipeline:
    """
    Constructs an end-to-end Pipeline:
    Preprocessor:
      - Numerical: SimpleImputer(median) -> StandardScaler()
      - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    Estimator:
      - LinearRegression()
    """
    num_subpipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_subpipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_subpipeline, numerical_cols),
            ("cat", cat_subpipeline, categorical_cols)
        ],
        remainder="drop"
    )
    
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    return model_pipeline


def train_model(
    cleaned_data_path: Path = DEFAULT_PROCESSED_DATA,
    train_indices_path: Path = DEFAULT_TRAIN_INDICES,
    model_output_path: Path = DEFAULT_MODEL_PATH,
    schema_output_path: Path = DEFAULT_SCHEMA_PATH
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains the Linear Regression pipeline strictly on training partition
    and exports fitted pipeline artifact and feature schema.
    """
    logger.info("Loading cleaned data from %s", cleaned_data_path)
    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_data_path}. Run preprocessing first.")
        
    df = pd.read_csv(cleaned_data_path)
    
    logger.info("Loading train indices from %s", train_indices_path)
    if not train_indices_path.exists():
        raise FileNotFoundError(f"Train indices not found at {train_indices_path}. Run preprocessing first.")
        
    train_indices_df = pd.read_csv(train_indices_path)
    
    # Filter strictly training rows
    if "original_index" in df.columns and "original_index" in train_indices_df.columns:
        train_df = df[df["original_index"].isin(train_indices_df["original_index"])].copy()
    else:
        train_df = df.iloc[train_indices_df.index].copy()
        
    logger.info("Training set size: %d rows x %d columns", train_df.shape[0], train_df.shape[1])
    
    # Identify feature types dynamically
    num_cols, cat_cols, excluded_cols = identify_column_types(train_df, target_col=TARGET_COL, id_col=ID_COL)
    
    ordered_features = num_cols + cat_cols
    logger.info("Identified %d numerical features: %s", len(num_cols), num_cols)
    logger.info("Identified %d categorical features: %s", len(cat_cols), cat_cols)
    logger.info("Excluded columns: %s", excluded_cols)
    
    X_train = train_df[ordered_features]
    y_train = train_df[TARGET_COL]
    
    # Build and fit pipeline
    pipeline = build_linear_regression_pipeline(num_cols, cat_cols)
    logger.info("Fitting LinearRegression pipeline on training set...")
    pipeline.fit(X_train, y_train)
    logger.info("Model fitting complete.")
    
    # Save model artifact
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    logger.info("Saved fitted model pipeline to %s", model_output_path)
    
    # Build comprehensive feature schema
    schema = {
        "project_name": "HouseWise",
        "author": "Person 1 (Data Preprocessing, EDA & Linear Regression)",
        "target": {
            "column_name": TARGET_COL,
            "unit": "INR Lakhs (1 Lakh = 100,000 INR)",
            "description": "Final property sale price in Lakhs"
        },
        "features": {
            "all_features_ordered": ordered_features,
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "excluded_columns": excluded_cols,
            "total_feature_count": len(ordered_features)
        },
        "split_configuration": {
            "train_ratio": 0.80,
            "test_ratio": 0.20,
            "random_state": 42,
            "train_row_count": len(train_df),
            "total_cleaned_rows": len(df)
        },
        "preprocessing_pipeline": {
            "numerical": "SimpleImputer(strategy='median') -> StandardScaler()",
            "categorical": "SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "estimator": "LinearRegression()"
        },
        "dataset_metadata": {
            "dataset_rows": len(df),
            "dataset_cols": len(df.columns)
        }
    }
    
    schema_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    logger.info("Saved feature schema to %s", schema_output_path)
    
    return pipeline, schema


if __name__ == "__main__":
    train_model()
