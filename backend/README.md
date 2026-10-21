# HouseWise — FastAPI Backend Service

Clean, high-performance REST API backend for **HouseWise: Indian House Price Prediction**.

---

## 📁 Project Structure

```
backend/
├── schemas.py         # Pydantic input validation, categorical options & response models
├── main.py            # FastAPI server, endpoints (/predict, /model-metrics), pipeline inference
├── test_api.py        # Automated 8-point verification test suite
├── requirements.txt   # Backend dependencies
└── README.md          # Setup & API documentation
```

---

## 🚀 Key Features

1. **`POST /predict`**:
   - Accepts 16 property features (`BHK`, `City`, `Carpet_Area_SqFt`, etc.).
   - Validates physical domain rules (e.g. `Carpet_Area_SqFt <= Super_Area_SqFt` and `Floor_Number <= Total_Floors`).
   - Runs inference using the trained `RandomForestRegressor` pipeline (`models/random_forest_model.joblib`) or `LinearRegression`.
   - Returns predicted price in INR Lakhs, human-friendly currency format (`₹ 4.82 Crore (482.19 Lakhs)`), and `mode: "real"`.
   - Automatic fallback to `mode: "mock"` if a model file is missing or still undergoing training.

2. **`GET /model-metrics`**:
   - Returns actual measured evaluation metrics (`MAE`, `RMSE`, `R²`) for Linear Regression and Random Forest from `reports/`.
   - Seamlessly compares all 4 algorithms (Linear Regression, Random Forest, XGBoost, CatBoost) for frontend chart visualization.

3. **`GET /health` & `GET /`**:
   - Quick health check verifying whether model `.joblib` files exist on disk.

4. **CORS Enabled**:
   - Configured for React + Vite dev server running on `http://localhost:5173`.

---

## 🛠️ Setup & Run

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Run the Development Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs (Swagger UI): `http://127.0.0.1:8000/docs`

### 3. Run the Automated Tests
```bash
python backend/test_api.py
```
*(Runs 8 test cases validating root, health, real predictions, mock mode, physical domain validations, metrics, and CORS).*
