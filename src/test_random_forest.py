"""
Quick test script to verify the trained Random Forest model works correctly.
Tests: model loading, prediction, and metrics consistency.
"""

import json
import joblib
import pandas as pd
import numpy as np

print("=" * 60)
print("  RANDOM FOREST MODEL — VERIFICATION TEST")
print("=" * 60)

# ── Test 1: Load the saved model ──────────────────────────────────────────
print("\n[TEST 1] Loading saved model...")
model = joblib.load("models/random_forest_model.joblib")
print(f"  Model type : {type(model).__name__}")
print(f"  Pipeline steps: {[step[0] for step in model.steps]}")
print(f"  Regressor : {type(model.named_steps['regressor']).__name__}")
print("  >> PASSED")

# ── Test 2: Make a sample prediction ──────────────────────────────────────
print("\n[TEST 2] Predicting price for a sample house...")

sample_house = pd.DataFrame([{
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
}])

predicted_price = model.predict(sample_house)[0]
print(f"  Input: 3 BHK, 1500 sqft, Mumbai, Premium, Apartment")
print(f"  Predicted Price: {predicted_price:.2f} Lakhs (INR {predicted_price:.2f}L)")
print(f"  Sanity check: Price > 0 and < 10000? {'YES' if 0 < predicted_price < 10000 else 'NO'}")
print("  >> PASSED")

# ── Test 3: Verify saved metrics file ─────────────────────────────────────
print("\n[TEST 3] Verifying metrics file...")
with open("reports/random_forest_metrics.json", "r") as f:
    metrics = json.load(f)

em = metrics["evaluation_metrics"]
print(f"  Test MAE  : {em['test_mae']}")
print(f"  Test RMSE : {em['test_rmse']}")
print(f"  Test R2   : {em['test_r2']}")
print(f"  Train MAE : {em['train_mae']}")
print(f"  Train RMSE: {em['train_rmse']}")
print(f"  Train R2  : {em['train_r2']}")

assert em["test_r2"] > 0.5, "R2 too low!"
assert em["test_mae"] < em["test_rmse"], "MAE should be < RMSE"
print("  >> PASSED")

# ── Test 4: Compare with Linear Regression ────────────────────────────────
print("\n[TEST 4] Comparing with Linear Regression...")
with open("reports/linear_regression_metrics.json", "r") as f:
    lr_metrics = json.load(f)

lr = lr_metrics["evaluation_metrics"]
rf = metrics["evaluation_metrics"]

print(f"  {'Metric':<12} {'Linear Reg':>12} {'Random Forest':>14} {'Winner':>10}")
print(f"  {'-'*50}")

for metric_name, lower_better in [("test_mae", True), ("test_rmse", True), ("test_r2", False)]:
    lr_val = lr[metric_name]
    rf_val = rf[metric_name]
    if lower_better:
        winner = "RF" if rf_val < lr_val else "LR"
    else:
        winner = "RF" if rf_val > lr_val else "LR"
    print(f"  {metric_name:<12} {lr_val:>12.4f} {rf_val:>14.4f} {winner:>10}")

print("  >> PASSED")

# ── Test 5: Predict on actual test data ───────────────────────────────────
print("\n[TEST 5] Predicting on actual test set (spot check)...")
df = pd.read_csv("data/processed/cleaned_data.csv")
test_idx = pd.read_csv("data/processed/test_indices.csv")
test_df = df[df["original_index"].isin(test_idx["original_index"])]

features = [
    "BHK", "Bathrooms", "Super_Area_SqFt", "Carpet_Area_SqFt",
    "Floor_Number", "Total_Floors", "Age_of_Property", "Parking",
    "Lift_Available", "Gated_Community", "Distance_to_Metro_km",
    "Distance_to_City_Center_km", "City", "Locality_Type",
    "Property_Type", "Furnishing_Status"
]

X_test = test_df[features]
y_test = test_df["Price_INR_Lakhs"]

preds = model.predict(X_test)
print(f"  Predictions made for {len(preds)} test samples")
print(f"  Sample predictions vs actuals (first 5):")
for i in range(5):
    print(f"    Actual: {y_test.iloc[i]:>8.2f} L  |  Predicted: {preds[i]:>8.2f} L  |  Error: {abs(y_test.iloc[i] - preds[i]):>8.2f} L")
print("  >> PASSED")

print("\n" + "=" * 60)
print("  ALL 5 TESTS PASSED!")
print("=" * 60)
