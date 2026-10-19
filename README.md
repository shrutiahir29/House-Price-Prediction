# HouseWise — Indian Urban House Price Prediction
## Person 1 Module: Data Preprocessing, Exploratory Data Analysis (EDA) & Linear Regression

Welcome to the **Person 1** module of the **HouseWise** machine learning system. This module establishes the data foundation, cleaning standards, exploratory insights, reproducible train/test partitions, and baseline Linear Regression model for the entire HouseWise engineering team.

---

## 1. Project Directory Structure

```text
HouseWise/
├── data/
│   ├── raw/
│   │   ├── train.csv                      # Original labelled Kaggle dataset (80,000 rows)
│   │   └── test.csv                       # Unlabelled test competition dataset (20,000 rows)
│   └── processed/
│       ├── cleaned_data.csv               # Cleaned labelled dataset preserving original row index
│       ├── train_indices.csv              # Exact row indices for 80% training partition (64,000 rows)
│       └── test_indices.csv               # Exact row indices for 20% test partition (16,000 rows)
├── notebooks/
│   ├── person1_linear_regression.ipynb   # Executed Jupyter notebook with complete EDA and modeling
│   └── generate_notebook.py              # Automated programmatic notebook generator
├── src/
│   ├── preprocessing.py                  # Ingestion, cleaning, domain checks, EDA plotting, and splitting
│   ├── train_linear_regression.py         # Leakage-free Scikit-Learn pipeline training & artifact export
│   └── evaluate_model.py                 # Metric evaluation (MAE, RMSE, R²) and automated pipeline verification
├── models/
│   └── linear_regression_model.joblib    # Fitted end-to-end pipeline ready for FastAPI deployment
├── reports/
│   ├── figures/
│   │   ├── price_distribution.png         # Raw & log1p price distribution plots
│   │   ├── missing_values_summary.png     # Missing value audit per feature
│   │   ├── city_price_comparison.png      # Ordered price comparison across 10 Indian cities
│   │   ├── correlation_heatmap.png        # Correlation matrix of numerical attributes vs price
│   │   ├── scatter_price_vs_features.png  # Area and distance vs price trend plots
│   │   └── boxplots_categorical_vs_price.png # Price vs locality, property type, furnishing, and BHK
│   ├── linear_regression_metrics.csv     # Model evaluation metrics in CSV format
│   ├── linear_regression_metrics.json    # Model evaluation metrics and interpretation in JSON format
│   ├── feature_schema.json               # Input feature specification contract for Person 2 & 3
│   └── cleaning_report.md                # Comprehensive data cleaning and outlier audit report
├── requirements.txt                      # Project dependency definitions
├── .gitignore                            # Git exclusion rules for environments, caches, and secrets
└── README.md                             # This documentation and integration guide
```

---

## 2. Dataset Overview & Verification

- **Source:** [Kaggle Indian Urban House Price Prediction Dataset](https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data)
- **Target Variable:** `Price_INR_Lakhs` (Verified float column; 1 Lakh = 100,000 INR).
- **Initial Dimensions:** 80,000 rows × 18 columns.
- **Identifier Column:** `Property_ID` (e.g. `PROP-005003`) is strictly excluded from feature sets to prevent memorization artifacts and data leakage.
- **Missingness:** 0 null values across all 80,000 records in raw dataset.
- **Outlier Decision:** No expensive luxury properties were deleted. In Indian metropolitan hubs (Mumbai, Delhi, Bangalore), properties trading above INR 10–33 Crore (1,000–3,334 Lakhs) are authentic market transactions. Deleting them would artificially truncate valuation for premium properties.

---

## 3. Five Key EDA Findings from Actual Data

1. **Substantial Positive Skewness:**
   - Target price median is **96.61 Lakhs** while mean is **142.61 Lakhs**, extending up to **3,334.92 Lakhs (INR 33.35 Crore)**.
   - 211 properties exceed INR 10 Crore; log transformation ($\log(1 + \text{Price})$) produces an approximately normal distribution.
2. **Geographic Price Hierarchy Across 10 Cities:**
   - **Mumbai** has the highest median price (**232.54 Lakhs**), followed by **Delhi** (**170.18 Lakhs**) and **Bangalore** (**113.97 Lakhs**).
   - **Jaipur** (**46.48 Lakhs**) and **Ahmedabad** (**56.73 Lakhs**) have the lowest median house prices. A Mumbai home has over **5.0x** the median price of a Jaipur home.
3. **Locality Type Tiering:**
   - **Premium:** Median = **230.64 Lakhs** (Mean = 284.77 Lakhs).
   - **Mid-Range:** Median = **93.75 Lakhs** (Mean = 115.45 Lakhs).
   - **Affordable:** Median = **45.55 Lakhs** (Mean = 56.52 Lakhs).
   - Moving from Affordable to Premium increases median price by over **5.06x**.
4. **Dominant Correlation with Built-up Area:**
   - `Super_Area_SqFt` ($r = 0.635$) and `Carpet_Area_SqFt` ($r = 0.624$) have the strongest linear correlation with price.
5. **Urban Center Proximity & Age Discount:**
   - `Age_of_Property` ($r = -0.108$) and `Distance_to_City_Center_km` ($r = -0.091$) consistently discount property valuations across all cities.

---

## 4. Reproducible Train/Test Partition

- **Split Ratio:** 80% Train (64,000 rows) / 20% Test (16,000 rows).
- **Random State:** `42` (`shuffle=True`).
- **Team Synchronization:** `data/processed/train_indices.csv` and `data/processed/test_indices.csv` contain the exact `original_index` and `Property_ID` values. Person 2 (Random Forest, CatBoost) and Person 3 (XGBoost) **must** filter on these indices to ensure identical evaluation sets.

---

## 5. Linear Regression Baseline Model

### Pipeline Architecture:
- **Numerical Features (12):** `BHK`, `Bathrooms`, `Super_Area_SqFt`, `Carpet_Area_SqFt`, `Floor_Number`, `Total_Floors`, `Age_of_Property`, `Parking`, `Lift_Available`, `Gated_Community`, `Distance_to_Metro_km`, `Distance_to_City_Center_km`.
  - Imputation: `SimpleImputer(strategy='median')`
  - Scaling: `StandardScaler()`
- **Categorical Features (4):** `City`, `Locality_Type`, `Property_Type`, `Furnishing_Status`.
  - Imputation: `SimpleImputer(strategy='most_frequent')`
  - Encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Estimator:** `LinearRegression()`
- **Target Leakage Prevention:** Preprocessing steps are strictly fit on the training partition only.

### Evaluation Metrics:

| Metric | Held-Out Test Set (20%) | Training Set (80%) | Interpretation |
| :--- | :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | **47.9403 Lakhs** | 47.7235 Lakhs | Average error magnitude (lower is better) |
| **RMSE (Root Mean Squared Error)** | **86.0255 Lakhs** | 84.2302 Lakhs | Penalizes large errors (lower is better) |
| **$R^2$ Score** | **0.6657** | 0.6710 | Explained variance (higher is better) |

*The close alignment between training ($R^2 = 0.6710$) and testing ($R^2 = 0.6657$) demonstrates that the baseline linear model generalizes consistently without overfitting.*

---

## 6. Team Integration Guide

### For Person 2 (FastAPI Backend Integration):
The saved model `models/linear_regression_model.joblib` is a complete pipeline that accepts **unprocessed raw feature dictionaries or DataFrames**. The FastAPI handler does not need to manually normalize or one-hot encode inputs.

```python
import joblib
import pandas as pd

# 1. Load pipeline at FastAPI startup
pipeline = joblib.load("models/linear_regression_model.joblib")

# 2. Example: Predict a single incoming property request
sample_input = {
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

# 3. Predict price
df = pd.DataFrame([sample_input])
predicted_price_lakhs = float(pipeline.predict(df)[0])
print(f"Predicted Price: {predicted_price_lakhs:.2f} Lakhs")
# Output: ~284.97 Lakhs
```

### For Person 3 (React Frontend Integration):
- Consult `reports/feature_schema.json` for exact field names, data types, and dropdown categorical values.
- Available Cities: `Bangalore`, `Chennai`, `Hyderabad`, `Delhi`, `Mumbai`, `Jaipur`, `Ahmedabad`, `Chandigarh`, `Kolkata`, `Pune`.
- Locality Types: `Affordable`, `Mid-Range`, `Premium`.
- Property Types: `Apartment`, `Builder Floor`, `Independent House`, `Villa`.
- Furnishing Status: `Furnished`, `Semi-Furnished`, `Unfurnished`.

---

## 7. How to Reproduce This Entire Pipeline

### Step 1: Set Up Python Environment
```bash
# Using standard venv (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Ensure Dataset is in Place
Place `train.csv` and `test.csv` in `data/raw/`:
```bash
mkdir -p data/raw
# Download from Kaggle and place in data/raw/train.csv and data/raw/test.csv
```

### Step 3: Run Data Preprocessing & EDA
```bash
python src/preprocessing.py
```
*Outputs: `data/processed/cleaned_data.csv`, `train_indices.csv`, `test_indices.csv`, 6 plots in `reports/figures/`, and `reports/cleaning_report.md`.*

### Step 4: Train Linear Regression Model
```bash
python src/train_linear_regression.py
```
*Outputs: `models/linear_regression_model.joblib` and `reports/feature_schema.json`.*

### Step 5: Evaluate Model & Run Automated Verification Tests
```bash
python src/evaluate_model.py
```
*Outputs: `reports/linear_regression_metrics.csv`, `reports/linear_regression_metrics.json`, and terminal verification report.*

### Step 6: (Optional) Re-generate Jupyter Notebook
```bash
python notebooks/generate_notebook.py
```

---

## 8. GitHub Collaboration & Handoff Commands

To stage, commit, and push your completed Person 1 deliverables to GitHub:

```bash
# Check git status
git status

# Stage Person 1 files (excluding virtual environments and caches via .gitignore)
git add src/ notebooks/ reports/ models/ data/processed/ requirements.txt .gitignore README.md

# Verify staged files
git status

# Create a clean, descriptive commit
git commit -m "feat(person1): complete data preprocessing, EDA, train/test split, and baseline linear regression pipeline"

# Push to your feature branch or main branch
git push origin <your-branch-name>
```
