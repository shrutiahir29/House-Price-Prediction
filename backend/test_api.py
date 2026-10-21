"""
HouseWise API — Verification Test Suite
Runs 10 automated tests against the FastAPI backend:
1. Root Endpoint (GET /)
2. Health Check & 4-Model Detection (GET /health)
3. Real Model Prediction with Random Forest (POST /predict)
4. Real Model Prediction with Linear Regression (POST /predict)
5. Real Model Prediction with Decision Tree (POST /predict)
6. Real Model Prediction with XGBoost (POST /predict)
7. Multi-Model Simultaneous Prediction (POST /predict-all)
8. Input Validation & Physical Domain Checks (422 Handlers)
9. All 4 Measured Model Metrics Comparison (GET /model-metrics)
10. Cross-Origin Resource Sharing (CORS Headers)
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def run_all_tests():
    print("=" * 65)
    print("  HOUSEWISE FASTAPI BACKEND — VERIFICATION TEST SUITE")
    print("=" * 65)

    # 1. Root
    print("\n[TEST 1] GET / (Root Endpoint)...")
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"
    print("  >> PASSED: API is online.")

    # 2. Health
    print("\n[TEST 2] GET /health (All 4 Model Files Verification)...")
    res = client.get("/health")
    assert res.status_code == 200
    models = res.json()["models_available"]
    assert models["random_forest"] is True, "Random Forest model file missing!"
    assert models["linear_regression"] is True, "Linear Regression model file missing!"
    assert models["decision_tree"] is True, "Decision Tree model file missing!"
    assert models["xgboost"] is True, "XGBoost model file missing!"
    print("  >> PASSED: All 4 models (RF, LR, DT, XGBoost) successfully verified on disk.")

    sample_input = {
        "BHK": 3,
        "Bathrooms": 2,
        "Super_Area_SqFt": 1500.0,
        "Carpet_Area_SqFt": 1100.0,
        "Floor_Number": 5,
        "Total_Floors": 12,
        "Age_of_Property": 5,
        "Parking": 1,
        "Lift_Available": 1,
        "Gated_Community": 1,
        "Distance_to_Metro_km": 2.5,
        "Distance_to_City_Center_km": 8.0,
        "City": "Mumbai",
        "Locality_Type": "Premium",
        "Property_Type": "Apartment",
        "Furnishing_Status": "Semi-Furnished",
        "model_name": "random_forest",
    }

    # 3. Predict with Random Forest
    print("\n[TEST 3] POST /predict (Real Random Forest Pipeline)...")
    res = client.post("/predict", json=sample_input)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "real"
    assert data["predicted_price_lakhs"] > 0
    print(f"  Predicted: {data['predicted_price_lakhs']} Lakhs ({data['formatted_price']})")
    print("  >> PASSED: Real Random Forest prediction working.")

    # 4. Predict with Linear Regression
    print("\n[TEST 4] POST /predict (Real Linear Regression Pipeline)...")
    lr_input = sample_input.copy()
    lr_input["model_name"] = "linear_regression"
    res = client.post("/predict", json=lr_input)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "real"
    print(f"  Predicted: {data['predicted_price_lakhs']} Lakhs ({data['formatted_price']})")
    print("  >> PASSED: Real Linear Regression prediction working.")

    # 5. Predict with Decision Tree
    print("\n[TEST 5] POST /predict (Real Decision Tree Pipeline)...")
    dt_input = sample_input.copy()
    dt_input["model_name"] = "decision_tree"
    res = client.post("/predict", json=dt_input)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "real"
    print(f"  Predicted: {data['predicted_price_lakhs']} Lakhs ({data['formatted_price']})")
    print("  >> PASSED: Real Decision Tree prediction working.")

    # 6. Predict with XGBoost
    print("\n[TEST 6] POST /predict (Real XGBoost Pipeline)...")
    xgb_input = sample_input.copy()
    xgb_input["model_name"] = "xgboost"
    res = client.post("/predict", json=xgb_input)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "real"
    print(f"  Predicted: {data['predicted_price_lakhs']} Lakhs ({data['formatted_price']})")
    print("  >> PASSED: Real XGBoost prediction working.")

    # 7. Predict-All Endpoint
    print("\n[TEST 7] POST /predict-all (Simultaneous Inference Across All 4 Models)...")
    res = client.post("/predict-all", json=sample_input)
    assert res.status_code == 200
    all_preds = res.json()["predictions"]
    for m, p in all_preds.items():
        print(f"  • {m:<18}: {p['predicted_price_lakhs']} Lakhs ({p['formatted_price']}) [Mode: {p['mode']}]")
        assert p["mode"] == "real"
    print("  >> PASSED: Predict-all executed real inference across all 4 models.")

    # 8. Validation Errors
    print("\n[TEST 8] Input Validation & Domain Guard Enforcement...")
    invalid_area = sample_input.copy()
    invalid_area["Carpet_Area_SqFt"] = 2500.0  # Carpet > Super Area
    res = client.post("/predict", json=invalid_area)
    assert res.status_code == 422
    print("  [✓] Blocked Carpet_Area > Super_Area (422 Unprocessable)")

    invalid_floor = sample_input.copy()
    invalid_floor["Floor_Number"] = 20
    invalid_floor["Total_Floors"] = 10  # Floor > Total Floors
    res = client.post("/predict", json=invalid_floor)
    assert res.status_code == 422
    print("  [✓] Blocked Floor_Number > Total_Floors (422 Unprocessable)")
    print("  >> PASSED: All boundary checks working.")

    # 9. Model Metrics Comparison
    print("\n[TEST 9] GET /model-metrics (All 4 Models Measured Comparison)...")
    res = client.get("/model-metrics")
    assert res.status_code == 200
    data = res.json()
    assert len(data["models"]) == 4
    for m in data["models"]:
        print(f"  • {m['model_name']:<42} | MAE: {m['mae']:>7.2f} | R²: {m['r2']:>6.4f} | Status: {m['status']}")
        assert m["status"] == "measured"
    print(f"  Champion Model: {data['best_model_name']}")
    print("  >> PASSED: All 4 models return measured metrics from disk.")

    # 10. CORS
    print("\n[TEST 10] CORS Headers Check (Frontend on localhost:5173)...")
    headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    res = client.options("/predict", headers=headers)
    assert res.status_code == 200
    assert "access-control-allow-origin" in res.headers
    print("  >> PASSED: CORS configured for frontend integration.")

    print("\n" + "=" * 65)
    print("  [ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY!]")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
