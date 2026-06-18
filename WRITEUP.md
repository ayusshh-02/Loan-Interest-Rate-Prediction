# Write-up — Loan Interest Rate Prediction

**Author:** Ayush Patel
**Deliverable:** End-to-end, runnable pipeline ([starter_code/loan_rate_prediction.py](starter_code/loan_rate_prediction.py)) + figures and metrics in [outputs/](outputs/).

---

## 1. How to run

The project runs in an isolated virtual environment.

```powershell
# from the exercise_1/ folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux

pip install -r requirements.txt
python starter_code/loan_rate_prediction.py
```

All artefacts are written to `outputs/`:

| File | Content |
|---|---|
| `interest_rate_dist.png` | Target distribution |
| `loan_type_dist.png` | Loan-type distribution |
| `avg_rate_by_type.png` | Average interest rate by loan type |
| `rate_drivers.png` | Credit score and income relationships |
| `correlation_heatmap.png` | Correlation matrix |
| `actual_vs_predicted.png` | Actual vs predicted values |
| `residual_distribution.png` | Residual analysis |
| `feature_importance.png` | Permutation feature importance |
| `metrics.csv` | Test-set metrics |

---

## 2. Part 1 — Exploratory Data Analysis

- **Shape:** 1,400 rows × 9 columns. `application_id` is a unique key (no duplicates).
- **Missing values:** `credit_score` (70), `annual_income` (69), `loan_term_months` (70) — roughly 5% each, missing-at-random in appearance.
- **Target (`interest_rate`):** Heavily right-skewed (skewness ≈ **4.1**). The skew is **not** organic — it is driven by two clusters of erroneous values: a floor of `0.05` (24 rows) and a ceiling of `50.0` (7 rows). These are classic sentinel / data-entry placeholders rather than real rates.
- **`loan_type` imbalance:** Strongly imbalanced — Personal (569) ≫ Home (414) > Auto (205) > Education (138) > Business (74). Business loans are ~8× rarer than Personal, so per-type estimates for Business are the least reliable.
- **Drivers:** `credit_score` shows a clear **negative** relationship with rate (higher score → lower rate), the dominant signal. `annual_income` shows a weaker negative relationship.

## 3. Part 2 — Data Cleaning & Preprocessing

**Impossible / erroneous values found and handled:**

| Field | Issue | Count | Action |
|---|---|---|---|
| `years_employed` | Negative values (e.g. −5) | 8 | Set to `NaN` → imputed |
| `credit_score` | Out of FICO range (300–850); `999` sentinels | 6 | Set to `NaN` → imputed |
| `annual_income` | Non-positive (≤ 0) | 8 | Set to `NaN` → imputed |
| `interest_rate` (target) | Out-of-domain (`0.05`, `50.0`) | 31 | **Rows dropped** |
| `interest_rate` (target) | IQR outliers beyond [5.64, 18.20] | 19 | **Rows dropped** |

**Decisions & justification:**

- **Outlier strategy for the target:** A two-stage approach — first a domain filter (valid APR band 3–35%) to remove obvious sentinels, then a standard 1.5×IQR guard on the cleaned target. The target is *dropped* rather than capped because it cannot be reliably imputed, and capping a placeholder like `50.0` to ~18% would invent a label. Net effect: 1,400 → **1,350** rows (50 dropped, all target-related). No feature rows are discarded.
- **Imputation strategy:** Numeric columns use **median** imputation (robust to skew/outliers); the categorical `loan_type` uses **most-frequent**. Crucially, imputation is fit **inside the modelling pipeline on the training fold only** — see leakage note below.
- **Encoding:** `loan_type` is nominal, so it is **one-hot encoded** rather than label-encoded (label encoding would impose a false ordinal ranking).
- **Scaling:** Numeric features are standardised (`StandardScaler`), which matters for the Ridge model.
- **Engineered feature:** `income_to_loan_ratio = annual_income / (loan_amount + 1)` as a proxy for repayment capacity.

## 4. Part 3 — Review & fixes to the starter code

The starter script contained three defects; each is fixed and annotated with `# FIX:` in the code.

| # | Bug | Consequence | Fix |
|---|---|---|---|
| 1 | `groupby().mean()` Series plotted without `reset_index()` | Bar chart x-axis labels silently wrong | Build an explicit DataFrame with `reset_index()` and plot with `seaborn.barplot`. |
| 2 | Credit-score filter kept **only** out-of-range scores (inverted comparison) | Would discard all valid rows and keep corrupt ones | Out-of-range scores set to `NaN` and imputed; valid rows retained. |
| 3 | `cross_val_score` run on a scaled NumPy array while `y` kept its original pandas index | Feature/target misalignment under fold shuffling → misleading CV scores | `X` and `y` rebuilt from the same `reset_index(drop=True)` frame; CV runs on the full pipeline. |

**Additional best-practice improvements beyond the listed bugs:**

- **Leakage removed:** the original cleaned/imputed/scaled the *entire* dataset before splitting. All transforms are now wrapped in a `Pipeline` + `ColumnTransformer` so they are fit on training data only and applied to the test fold — the correct, leakage-free setup.
- Switched from `LabelEncoder` (ordinal, wrong for nominal data) to `OneHotEncoder`.
- Cross-validation now uses an explicit shuffled `KFold` and scores the **whole pipeline**, so preprocessing is re-fit per fold.
- Feature importance switched to **permutation importance** (model-agnostic, evaluated on held-out data) instead of impurity-based importances, which are biased toward high-cardinality features.
- Code refactored into functions with a single `main()`, headless matplotlib backend, path-independent execution, and all artefacts saved to `outputs/`.

## 5. Part 4 — Model Building & Evaluation

Two regression models were trained and evaluated on a held-out 20% test set.

| Model                | MAE       | RMSE      | R²        |
| -------------------- | --------- | --------- | --------- |
| **Ridge Regression** | **1.190** | **1.507** | **0.485** |
| Random Forest        | 1.329     | 1.660     | 0.375     |

### Model Selection

The **Ridge Regression** model achieved the best performance across all evaluation metrics and was selected as the final model.

An interesting observation is that the linear Ridge model outperformed the more complex Random Forest model. This suggests that the relationship between the available borrower attributes and the assigned interest rate is largely linear, with credit score acting as the dominant driver. The Random Forest model appears to capture additional noise rather than meaningful non-linear patterns.

### Cross-Validation Results

A 5-fold cross-validation was performed using the Ridge Regression model.

| Metric             | Value     |
| ------------------ | --------- |
| Mean CV R²         | **0.525** |
| Standard Deviation | **0.041** |

Fold scores:

```text
[0.4853, 0.5953, 0.5068, 0.4914, 0.5442]
```

The relatively low standard deviation indicates that model performance is stable across different data splits and is not dependent on a particularly favorable train-test split.

### Error Interpretation

The final model achieved an MAE of approximately **1.19 percentage points**, meaning that the predicted interest rate is typically within ±1.2 percentage points of the actual assigned rate.

This level of accuracy is suitable for supporting underwriting decisions and pricing recommendations, although it may not be sufficient for fully automated rate assignment without additional business rules and validation controls.

### Feature Importance

Permutation feature importance on the best-performing model produced the following ranking:

| Feature                | Importance |
| ---------------------- | ---------- |
| `credit_score`         | 0.849      |
| `loan_type`            | 0.174      |
| `annual_income`        | 0.034      |
| `years_employed`       | 0.025      |
| `income_to_loan_ratio` | 0.010      |
| `applicant_age`        | 0.006      |
| `loan_term_months`     | -0.001     |
| `loan_amount`          | -0.001     |

The results show that **credit score overwhelmingly dominates the prediction process**, accounting for the vast majority of predictive power. Loan type contributes a meaningful secondary signal, while the remaining variables provide relatively minor incremental value.

The near-zero importance of loan amount and loan term suggests that, within this dataset, these variables contribute little additional information once credit score and loan type are known.

## 6. Part 5 — Business Interpretation

### Which applicants are likely to receive the highest rates?

The model indicates that applicants with **lower credit scores** are substantially more likely to receive higher interest rates. Credit score is by far the most influential feature in the model and acts as the primary measure of borrower risk.

Loan type also contributes to pricing decisions. Certain categories, particularly unsecured lending products such as Personal loans, tend to receive higher rates than lower-risk categories such as Home loans.

Annual income and employment history have a measurable but comparatively small influence on the final rate assignment.

### Does this seem fair?

From a risk-based lending perspective, the model behavior is economically reasonable because higher-risk borrowers are assigned higher rates.

However, credit scores can indirectly reflect broader socioeconomic factors and historical lending patterns. Therefore, before deployment, the model should be audited for potential disparate impact across protected groups even though protected attributes are not explicitly included in the training data.

### Additional data that would improve the model

The current dataset contains only a limited set of underwriting variables. Model performance could likely be improved by incorporating:

* Debt-to-income ratio (DTI)
* Existing liabilities and open credit lines
* Delinquency and repayment history
* Previous defaults or bankruptcies
* Home ownership status
* Employment type and job stability
* Loan-to-value ratio for secured loans
* Requested versus approved loan amount

These variables would provide a more complete representation of borrower risk and would likely improve predictive performance beyond the current R² level of approximately 0.5.

### Production Monitoring

If deployed, the model should be monitored through:

1. **Data Quality Monitoring**

   * Missing value rates
   * Invalid credit scores
   * Negative employment duration
   * Unexpected interest-rate values

2. **Data Drift Monitoring**

   * Population Stability Index (PSI)
   * Distribution changes in credit score, income, and loan type

3. **Model Performance Monitoring**

   * MAE
   * RMSE
   * R²
   * Prediction distributions

4. **Fairness Monitoring**

   * Segment-level performance analysis
   * Bias and disparate-impact assessments

5. **Model Governance**

   * Scheduled retraining
   * Champion–challenger model evaluation
   * Versioning and audit logging

## 7. Submission checklist

- [x] `loan_rate_prediction.py` — complete and runnable end-to-end (in a venv)
- [x] All plots saved as `.png` (in `outputs/`)
- [x] `WRITEUP.md` covering decisions and findings
- [x] `requirements.txt` for reproducible setup
