# HouseWise Data Cleaning & Preprocessing Report
**Module:** Person 1 (Data Preprocessing, EDA & Linear Regression)  
**Project:** HouseWise — Indian Urban House Price Prediction  
**Dataset Source:** Kaggle Indian Urban House Price Prediction  
**Target Variable:** `Price_INR_Lakhs` (Sale price in Indian Lakhs, 1 Lakh = 100,000 INR)  

---

## 1. Executive Summary & Row Counts

| Metric | Value |
| :--- | :--- |
| **Initial Raw Records** | `80,000` |
| **Final Cleaned Records** | `80,000` |
| **Total Rows Removed** | `0` |
| **Cleaned Features Count** | `16` predictive features |
| **Excluded Identifier Columns** | `Property_ID, Price_INR_Lakhs, original_index` |
| **Training Split Partition (80%)** | `64,000` rows |
| **Held-Out Test Partition (20%)** | `16,000` rows |
| **Random State for Split** | `42` |

---

## 2. Preprocessing & Cleaning Decisions

### A. Target Variable Verification & Integrity
- **Verified Column Name:** `Price_INR_Lakhs`
- **Conversion:** Parsed to standard 64-bit float using `pd.to_numeric`.
- **Validation:** Checked for negative, zero, or null price values.
- **Result:** Found `0` missing or invalid target values. All rows contained valid positive prices.

### B. Duplicate Detection
- Checked for exact duplicate rows across all predictive features.
- Found `0` exact duplicate rows.
- No rows required dropping due to duplication.

### C. Identifier Columns
- `Property_ID` (e.g., `PROP-005003`) was strictly excluded from predictive features to prevent data leakage and memorization artifacts.
- The raw dataset row position is saved as `original_index` in the cleaned dataset to guarantee 100% split synchronization across all team models.

### D. Missing Values Strategy
- **Raw Missingness:** The raw Kaggle dataset is fully populated (0 null entries across all 80,000 training rows).
- **Production-Grade Resilience:** Implemented standard scikit-learn transformers inside the pipeline:
  - Numerical features: `SimpleImputer(strategy='median')`
  - Categorical features: `SimpleImputer(strategy='most_frequent')`
  - Categorical encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Leakage Prevention:** All imputers and encoders are fitted **strictly on the 80% training partition** and applied without refitting to the test set or new incoming inference inputs.

### E. Domain Sanity & Outlier Audit
- **Floor Number vs Total Floors:** Validated that `Floor_Number <= Total_Floors` across 100% of rows (`0` violations).
- **Carpet Area vs Super Area:** Validated that `Carpet_Area_SqFt <= Super_Area_SqFt` across 100% of rows (`0` violations).
- **Outlier Assessment:**
  - Median House Price: `96.6 Lakhs` (Q1: `54.35 L`, Q3: `176.71 L`)
  - Interquartile Range (IQR) Upper Bound: `360.26 Lakhs`
  - Records above IQR bound: `5,355`
  - Properties > INR 10 Crore (1,000 Lakhs): `211`
  - Maximum Price: `3334.92 Lakhs` (INR 33.35 Crore)
  - **Decision:** As specified in the senior engineering requirements, **no expensive properties were deleted**. In Indian metropolitan markets (Mumbai, South Delhi, Bangalore central), luxury penthouses and large independent bungalows legitimately trade between 10 Cr and 35+ Cr. Removing them would cause artificial truncation and severely impair real-world luxury valuation.

---

## 3. Feature Categorization

### Numerical Features (12 features)
`BHK`, `Bathrooms`, `Super_Area_SqFt`, `Carpet_Area_SqFt`, `Floor_Number`, `Total_Floors`, `Age_of_Property`, `Parking`, `Lift_Available`, `Gated_Community`, `Distance_to_Metro_km`, `Distance_to_City_Center_km`

### Categorical Features (4 features)
`City`, `Locality_Type`, `Property_Type`, `Furnishing_Status`

---

## 4. Reproducibility & Team Handoff Files

1. `data/processed/cleaned_data.csv`: Cleaned dataset preserving `original_index`.
2. `data/processed/train_indices.csv`: 64,000 original indices for training (80%).
3. `data/processed/test_indices.csv`: 16,000 original indices for testing (20%).
4. All team members (Person 2: Random Forest / CatBoost / FastAPI; Person 3: XGBoost / React) must filter using these exact index files to ensure zero distribution divergence.
