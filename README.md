# Loan Interest Rate Prediction

## Overview

This project builds an end-to-end machine learning pipeline to predict the interest rate offered to loan applicants based on their financial profile and loan characteristics.

The solution covers:

* Exploratory Data Analysis (EDA)
* Data Cleaning & Validation
* Feature Engineering
* Leakage-safe Preprocessing
* Regression Modeling
* Cross-Validation
* Feature Importance Analysis
* Business Interpretation

---

## Repository Structure

```text
exercise_1/
│
├── data/
│   └── loan_applications.csv
│
├── codes/
│   └── loan_rate_prediction.py
│
├── outputs/
│   ├── interest_rate_dist.png
│   ├── loan_type_dist.png
│   ├── avg_rate_by_type.png
│   ├── rate_drivers.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── residual_distribution.png
│   ├── feature_importance.png
│   └── metrics.csv
│
├── WRITEUP.md
├── requirements.txt
└── README.md
```

---

## Approach

### 1. Exploratory Data Analysis

* Dataset shape and schema validation
* Missing-value analysis
* Interest-rate distribution analysis
* Loan-type distribution analysis
* Credit score and income relationship analysis
* Correlation heatmap

### 2. Data Cleaning

The following data-quality issues were identified and addressed:

| Field          | Issue                    | Action               |
| -------------- | ------------------------ | -------------------- |
| years_employed | Negative values          | Converted to missing |
| credit_score   | Outside valid FICO range | Converted to missing |
| annual_income  | Non-positive values      | Converted to missing |
| interest_rate  | Invalid sentinel values  | Removed              |
| interest_rate  | Extreme outliers         | Removed              |

### 3. Feature Engineering

Created:

* income_to_loan_ratio

This feature acts as a proxy for repayment capacity.

### 4. Preprocessing

Implemented using:

* Pipeline
* ColumnTransformer

Transformations:

* Median imputation for numeric features
* Most-frequent imputation for categorical features
* One-hot encoding for loan type
* Standard scaling for numerical variables

This design prevents data leakage by fitting all preprocessing steps on training data only.

### 5. Modeling

Models evaluated:

1. Ridge Regression
2. Random Forest Regressor

Evaluation metrics:

* MAE
* RMSE
* R²

Cross-validation:

* 5-Fold Cross Validation

### 6. Feature Importance

Permutation importance was used to identify the most influential features.

Key finding:

* Credit score is the strongest predictor of interest rate.

### 7. Business Insights

Applicants with lower credit scores receive significantly higher interest rates.

Additional variables such as debt-to-income ratio, delinquency history, and existing liabilities would likely improve predictive performance.

---

## Results

| Model            | MAE   | RMSE  | R²    |
| ---------------- | ----- | ----- | ----- |
| Ridge Regression | 1.190 | 1.507 | 0.485 |
| Random Forest    | 1.329 | 1.660 | 0.375 |

Best Model: **Ridge Regression**

Cross Validation:

* Mean R²: 0.525
* Std: 0.041

---

## Setup

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python codes/loan_rate_prediction.py
```

---

## Assumptions

* Valid credit-score range: 300–850
* Valid interest-rate range: 3–35%
* Missing values are assumed to be Missing At Random (MAR)
* Interest-rate outliers represent data-entry errors rather than true observations

> **Note:** For detailed analysis, methodology, assumptions, model evaluation, and business insights, please refer to `WRITEUP.md`.
---

## Author

Ayush Patel
