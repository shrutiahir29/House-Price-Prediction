"""
HouseWise: Indian House Price Prediction — FastAPI Backend
Clean, production-ready REST API connecting trained ML pipelines to the React frontend.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import HouseInput, MetricsResponse, ModelMetricItem, PredictionResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("housewise_api")

# ── Paths & Schema Constants ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATHS = {
    "random_forest": MODELS_DIR / "random_forest_model.joblib",
    "linear_regression": MODELS_DIR / "linear_regression_model.joblib",
    "xgboost": MODELS_DIR / "xgboost_model.joblib",
    "catboost": MODELS_DIR / "catboost_model.joblib",
}

ALL_FEATURES = [
    "BHK", "Bathrooms", "Super_Area_SqFt", "Carpet_Area_SqFt",
    "Floor_Number", "Total_Floors", "Age_of_Property", "Parking",
    "Lift_Available", "Gated_Community", "Distance_to_Metro_km",
    "Distance_to_City_Center_km", "City", "Locality_Type",
    "Property_Type", "Furnishing_Status",
]

# In-memory pipeline cache
_pipeline_cache: Dict[str, Any] = {}


# ── Helper Functions ─────────────────────────────────────────────────────────
def format_inr(price_lakhs: float) -> str:
    """Formats price nicely in Indian Lakhs or Crores."""
    if price_lakhs >= 100.0:
        return f"₹ {price_lakhs / 100.0:.2f} Crore ({price_lakhs:.2f} Lakhs)"
    return f"₹ {price_lakhs:.2f} Lakhs"


def get_model_pipeline(model_name: str):
    """Loads and caches trained Scikit-learn pipelines from disk."""
    if model_name in _pipeline_cache:
        return _pipeline_cache[model_name]

    path = MODEL_PATHS.get(model_name)
    if path and path.exists():
        try:
            logger.info("Loading model artifact: %s", path)
            pipeline = joblib.load(path)
            _pipeline_cache[model_name] = pipeline
            return pipeline
        except Exception as e:
            logger.error("Error loading model %s: %s", path, e)
    return None


def calculate_mock_price(features: dict, model_name: str) -> float:
    """Heuristic fallback calculation when a model file is not yet available."""
    city_weights = {"Mumbai": 2.6, "Delhi": 1.9, "Bangalore": 1.7, "Pune": 1.3,
                    "Hyderabad": 1.4, "Chennai": 1.3, "Kolkata": 1.1, "Chandigarh": 1.2,
                    "Ahmedabad": 1.0, "Jaipur": 0.9}
    loc_weights = {"Premium": 1.5, "Mid-Range": 1.1, "Affordable": 0.8}
    prop_weights = {"Villa": 1.6, "Independent House": 1.3, "Apartment": 1.0, "Builder Floor": 0.95}

    area = features.get("Super_Area_SqFt", 1200)
    base = area * 0.055
    price = (
        base
        * city_weights.get(features.get("City", "Mumbai"), 1.2)
        * loc_weights.get(features.get("Locality_Type", "Mid-Range"), 1.0)
        * prop_weights.get(features.get("Property_Type", "Apartment"), 1.0)
        * max(0.85, 1.0 - (features.get("Distance_to_Metro_km", 3.0) * 0.02))
        * max(0.75, 1.0 - (features.get("Age_of_Property", 5) * 0.01))
    )
    return round(float(price), 2)


# ── FastAPI App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="HouseWise API",
    description="Indian House Price Prediction & Model Benchmarking Service",
    version="1.0.0",
)

# CORS Middleware (allows React Vite frontend on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "project": "HouseWise — Indian House Price Prediction",
        "status": "online",
        "docs": "/docs",
        "endpoints": ["POST /predict", "GET /model-metrics", "GET /health"],
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "models_available": {
            name: (path.exists()) for name, path in MODEL_PATHS.items()
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: HouseInput):
    """
    Predicts house price in INR Lakhs.
    Uses trained Random Forest or Linear Regression pipeline; falls back to mock if file is missing.
    """
    model_name = payload.model_name or "random_forest"
    features_dict = payload.model_dump(exclude={"model_name"})
    pipeline = get_model_pipeline(model_name)

    if pipeline is not None:
        try:
            input_df = pd.DataFrame([features_dict])[ALL_FEATURES]
            predicted_val = float(pipeline.predict(input_df)[0])
            price_lakhs = max(5.0, round(predicted_val, 2))
            mode = "real"
            note = f"Predicted using trained {model_name.replace('_', ' ').title()} pipeline."
        except Exception as e:
            logger.error("Inference failure: %s", e)
            price_lakhs = calculate_mock_price(features_dict, model_name)
            mode = "mock"
            note = f"Inference failed ({str(e)}). Fallen back to simulated estimation."
    else:
        price_lakhs = calculate_mock_price(features_dict, model_name)
        mode = "mock"
        note = f"Model artifact '{model_name}' not yet trained. Returned simulated estimate."

    return PredictionResponse(
        predicted_price_lakhs=price_lakhs,
        formatted_price=format_inr(price_lakhs),
        model_used=model_name,
        mode=mode,
        features_received=features_dict,
        note=note,
    )


@app.get("/model-metrics", response_model=MetricsResponse, tags=["Evaluation"])
def get_model_metrics():
    """
    Returns actual MAE, RMSE, and R² metrics for all 4 models.
    Loads measured reports from disk, providing marked mock placeholders for pending models.
    """
    models = []

    # 1. Linear Regression (Person 1)
    lr_file = REPORTS_DIR / "linear_regression_metrics.json"
    if lr_file.exists():
        with open(lr_file, "r", encoding="utf-8") as f:
            em = json.load(f).get("evaluation_metrics", {})
            models.append(
                ModelMetricItem(
                    model_id="linear_regression",
                    model_name="Linear Regression (Baseline Pipeline)",
                    mae=round(float(em.get("test_mae", 47.9403)), 4),
                    rmse=round(float(em.get("test_rmse", 86.0255)), 4),
                    r2=round(float(em.get("test_r2", 0.6657)), 4),
                    train_mae=round(float(em.get("train_mae", 47.7235)), 4),
                    train_rmse=round(float(em.get("train_rmse", 84.2302)), 4),
                    train_r2=round(float(em.get("train_r2", 0.671)), 4),
                    status="measured",
                )
            )

    # 2. Random Forest (Person 2)
    rf_file = REPORTS_DIR / "random_forest_metrics.json"
    if rf_file.exists():
        with open(rf_file, "r", encoding="utf-8") as f:
            em = json.load(f).get("evaluation_metrics", {})
            models.append(
                ModelMetricItem(
                    model_id="random_forest",
                    model_name="Random Forest Regressor (Tuned Pipeline)",
                    mae=round(float(em.get("test_mae", 32.9443)), 4),
                    rmse=round(float(em.get("test_rmse", 68.9376)), 4),
                    r2=round(float(em.get("test_r2", 0.7853)), 4),
                    train_mae=round(float(em.get("train_mae", 24.409)), 4),
                    train_rmse=round(float(em.get("train_rmse", 53.9469)), 4),
                    train_r2=round(float(em.get("train_r2", 0.865)), 4),
                    status="measured",
                )
            )

    # 3. XGBoost (Person 3 / Placeholder)
    xgb_file = REPORTS_DIR / "xgboost_metrics.json"
    if xgb_file.exists():
        with open(xgb_file, "r", encoding="utf-8") as f:
            em = json.load(f).get("evaluation_metrics", {})
            models.append(ModelMetricItem(
                model_id="xgboost", model_name="XGBoost Regressor (Tuned)",
                mae=round(float(em.get("test_mae", 0)), 4), rmse=round(float(em.get("test_rmse", 0)), 4),
                r2=round(float(em.get("test_r2", 0)), 4), status="measured"
            ))
    else:
        models.append(ModelMetricItem(
            model_id="xgboost", model_name="XGBoost Regressor (Pending / Person 3)",
            mae=30.15, rmse=64.20, r2=0.8120, train_mae=21.30, train_rmse=48.50, train_r2=0.8910,
            status="simulated_mock"
        ))

    # 4. CatBoost (Placeholder)
    cb_file = REPORTS_DIR / "catboost_metrics.json"
    if cb_file.exists():
        with open(cb_file, "r", encoding="utf-8") as f:
            em = json.load(f).get("evaluation_metrics", {})
            models.append(ModelMetricItem(
                model_id="catboost", model_name="CatBoost Regressor (Tuned)",
                mae=round(float(em.get("test_mae", 0)), 4), rmse=round(float(em.get("test_rmse", 0)), 4),
                r2=round(float(em.get("test_r2", 0)), 4), status="measured"
            ))
    else:
        models.append(ModelMetricItem(
            model_id="catboost", model_name="CatBoost Regressor (Pending)",
            mae=29.80, rmse=62.90, r2=0.8210, train_mae=20.80, train_rmse=47.10, train_r2=0.8980,
            status="simulated_mock"
        ))

    # Mark top model
    measured = [m for m in models if m.status == "measured"]
    best = max(measured, key=lambda m: m.r2) if measured else max(models, key=lambda m: m.r2)
    for m in models:
        m.is_best = (m.model_id == best.model_id)

    return MetricsResponse(
        project="HouseWise",
        target_unit="INR Lakhs",
        models=models,
        best_model_id=best.model_id,
        best_model_name=best.model_name,
        summary=(
            f"Top measured model is '{best.model_name}' with R² score of {best.r2:.4f} "
            f"and MAE of {best.mae:.2f} Lakhs."
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
