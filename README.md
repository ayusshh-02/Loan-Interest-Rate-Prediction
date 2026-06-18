# AI Analyst Exercise — Loan Interest Rate Prediction

> **Time estimate:** 2–3 hours  
> **Submission:** Share a GitHub repo with your code, outputs, and a brief write-up.

---

## Background

A consumer lending company wants to move away from manual underwriting decisions and build a data-driven model that predicts the **interest rate** to offer an applicant, based on their financial profile and loan characteristics.

You have been given a dataset of **1,400 historical loan applications** along with the interest rate that was eventually assigned. Your task is to explore this data, prepare it for modelling, build a predictive model, and communicate your findings.

---

## Repository Structure

```
exercise_1/
├── data/
│   └── loan_applications.csv     ← Dataset (do not modify)
├── codes/
│   └── loan_rate_prediction.py   ← Scripts
└── README.md                     ← This file
```

---

## Dataset Description

**File:** `data/loan_applications.csv`

| Column | Type | Description |
|---|---|---|
| `application_id` | string | Unique application identifier |
| `applicant_age` | int | Age of applicant |
| `years_employed` | int | Current continuous employment duration |
| `loan_type` | string | Type of loan (Personal, Home, Auto, Education, Business) |
| `credit_score` | int | Bureau credit score at time of application |
| `annual_income` | int | Self-reported gross annual income (USD) |
| `loan_amount` | int | Requested loan amount (USD) |
| `loan_term_months` | int | Requested repayment term in months |
| `interest_rate` | float | Assigned interest rate (%) — **target variable** |

> The dataset reflects real-world messiness: some fields have missing values, some records contain data entry errors, and the loan type distribution is not uniform.

---

## Tasks

### Part 1 — Exploratory Data Analysis
1. Summarise the dataset: shape, column types, missing values, and descriptive statistics.
2. Plot and describe the distribution of the target variable (`interest_rate`). Is it skewed? Are there anomalies?
3. Visualise the distribution of `loan_type` and comment on any imbalance you notice.
4. Explore how `credit_score` and `annual_income` relate to `interest_rate`. Use appropriate charts.

### Part 2 — Data Cleaning & Preprocessing
1. Identify and handle impossible or erroneous values (consider valid ranges for each field).
2. Detect and remove or cap outliers in `interest_rate`. Document your approach and threshold choice.
3. Impute missing values — justify the strategy used for each column.
4. Encode categorical variables appropriately and scale numeric features.

### Part 3 — Review & Improve the Starter Code
Open `starter_code/loan_rate_prediction.py`. It provides a skeleton for the full pipeline.

Go through the code carefully. You may find areas where the logic does not work as intended, where results could be misleading, or where the implementation does not follow best practice. Fix anything you identify, explain your changes in comments or your write-up, and extend the script to complete all tasks.

### Part 4 — Model Building & Evaluation
1. Train at least **two regression models**.
2. Report **MAE**, **RMSE**, and **R²** on the test set.
3. Perform **5-fold cross-validation** on your best model and report mean ± std.
4. Plot **feature importances** and discuss which features matter most.

### Part 5 — Business Interpretation *(encouraged)*
- Which applicant profiles are likely to receive the highest rates? Does this seem fair?
- What data would you add to improve the model further?
- How would you monitor this model once it is in production?

---

## Environment Setup

Run inside an isolated virtual environment:

```powershell
# from the exercise_1/ folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux

pip install -r requirements.txt
python starter_code/loan_rate_prediction.py
```

Outputs (figures + `metrics.csv`) are written to `outputs/`. See [WRITEUP.md](WRITEUP.md) for the full analysis.

---

## Submission Checklist

- [ ] `loan_rate_prediction.py` — complete and runnable end-to-end
- [ ] All plots saved as `.png`
- [ ] `WRITEUP.md` (or inline comments) covering your decisions and findings
