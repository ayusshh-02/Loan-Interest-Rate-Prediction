"""
AI Analyst Exercise — Exercise 1
Loan Interest Rate Prediction
=================================================================
End-to-end, production-minded regression pipeline that predicts the
interest rate offered to a loan applicant from their financial profile.

Author : Ayush Patel
Run    : python starter_code/loan_rate_prediction.py
Output : ./outputs/*.png  (figures) and ./outputs/metrics.csv (scores)

Design notes
------------
* The script is organised into small, testable functions and a single
  `main()` entry point so it runs cleanly end-to-end.
* All file paths are resolved relative to this file, so the script works
  regardless of the current working directory.
* Three bugs that were present in the original starter skeleton are fixed
  and annotated inline with `# FIX:` so the reasoning is auditable.
* A scikit-learn `Pipeline` + `ColumnTransformer` is used so that all
  imputation/scaling/encoding is fit on the TRAINING data only and applied
  consistently to the test fold — this prevents data leakage, which the
  original "clean-everything-up-front" approach was prone to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend so the script runs without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Paths & global config
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "loan_applications.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Domain-valid ranges used to flag impossible / data-entry-error values.
VALID_RANGES = {
    "applicant_age": (18, 100),
    "years_employed": (0, 80),
    "credit_score": (300, 850),    # standard FICO range
    "annual_income": (1, None),    # must be strictly positive
    "loan_amount": (1, None),
    "loan_term_months": (1, None),
    "interest_rate": (3.0, 35.0),  # plausible consumer-lending APR band
}

sns.set_theme(style="whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw loan-applications dataset."""
    df = pd.read_csv(path)
    print("=" * 70)
    print("1. DATA LOADED")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nFirst rows:")
    print(df.head())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    """Summarise the data and save the four required EDA figures."""
    print("\n" + "=" * 70)
    print("2. EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDescriptive statistics:")
    print(df.describe(include="all").T)

    # --- 2a. Target distribution ----------------------------------------------
    plt.figure(figsize=(8, 4))
    sns.histplot(df["interest_rate"], bins=40, kde=True, color="#4C72B0")
    plt.title("Interest Rate Distribution (raw)")
    plt.xlabel("Interest Rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "interest_rate_dist.png", dpi=120)
    plt.close()

    skew = df["interest_rate"].skew()
    print(f"\nInterest-rate skewness: {skew:.3f}")
    print(
        "Note: sentinel/erroneous values (~0.05 and 50.0) create a long right "
        "tail; these are handled in cleaning."
    )

    # --- 2b. Loan-type distribution -------------------------------------------
    loan_counts = df["loan_type"].value_counts()
    print("\nLoan-type distribution:")
    print(loan_counts)

    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df,
        x="loan_type",
        order=loan_counts.index,
        hue="loan_type",
        palette="viridis",
        legend=False,
    )
    plt.title("Loan Type Distribution")
    plt.xlabel("Loan Type")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loan_type_dist.png", dpi=120)
    plt.close()

    # --- 2c. Average rate by loan type ----------------------------------------
    # FIX (Bug 1): the original grouped Series was plotted without resetting the
    # index, which produced mislabeled / silently-failing bar charts. Using an
    # explicit DataFrame via reset_index() gives clean, correct axis labels.
    avg_rate_by_type = (
        df.groupby("loan_type")["interest_rate"].mean().reset_index()
        .sort_values("interest_rate", ascending=False)
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=avg_rate_by_type,
        x="loan_type",
        y="interest_rate",
        hue="loan_type",
        palette="rocket",
        legend=False,
    )
    plt.title("Average Interest Rate by Loan Type")
    plt.xlabel("Loan Type")
    plt.ylabel("Avg Interest Rate (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "avg_rate_by_type.png", dpi=120)
    plt.close()

    # --- 2d. Drivers: credit score & income vs rate ---------------------------
    _, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(
        data=df, x="credit_score", y="interest_rate", alpha=0.4,
        ax=axes[0], color="#55A868",
    )
    axes[0].set_title("Credit Score vs Interest Rate")
    sns.scatterplot(
        data=df, x="annual_income", y="interest_rate", alpha=0.4,
        ax=axes[1], color="#C44E52",
    )
    axes[1].set_title("Annual Income vs Interest Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rate_drivers.png", dpi=120)
    plt.close()


    # --- 2e. Correlation Heatmap ---------------------------------------------
    numeric_df = df.select_dtypes(include=np.number)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=120
    )

    plt.close()

    
    print("\nEDA figures saved to:", OUTPUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEANING (targets only — features cleaned inside the pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the cleaning steps that must happen *before* train/test split:

    * Convert impossible feature values to NaN so they are imputed later
      (imputation itself is done inside the modelling Pipeline to avoid leakage).
    * Drop rows whose TARGET is erroneous. The target cannot be imputed, so
      sentinel values (~0.05 and 50.0) and IQR outliers are removed.
    """
    print("\n" + "=" * 70)
    print("3. DATA CLEANING")
    print("=" * 70)
    df = df.copy()
    start_rows = len(df)

    # --- 3a. Flag impossible feature values as NaN ----------------------------
    # years_employed cannot be negative.
    n_neg_emp = int((df["years_employed"] < 0).sum())
    df.loc[df["years_employed"] < 0, "years_employed"] = np.nan

    # FIX (Bug 2): the original filter KEPT only out-of-range credit scores
    # (inverted comparison). We instead null out-of-range scores [300, 850]
    # so they get imputed rather than corrupting the model.
    lo, hi = VALID_RANGES["credit_score"]
    bad_credit = (df["credit_score"] < lo) | (df["credit_score"] > hi)
    n_bad_credit = int(bad_credit.sum())
    df.loc[bad_credit, "credit_score"] = np.nan

    # Non-positive income is impossible.
    n_bad_income = int((df["annual_income"] <= 0).sum())
    df.loc[df["annual_income"] <= 0, "annual_income"] = np.nan

    print(f"Flagged as missing -> years_employed<0: {n_neg_emp}, "
          f"credit_score out-of-range: {n_bad_credit}, income<=0: {n_bad_income}")

    # --- 3b. Remove erroneous TARGET rows -------------------------------------
    t_lo, t_hi = VALID_RANGES["interest_rate"]
    domain_mask = df["interest_rate"].between(t_lo, t_hi)
    n_domain = int((~domain_mask).sum())
    df = df[domain_mask].copy()

    # Secondary IQR guard on the now-cleaned target.
    q1, q3 = df["interest_rate"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_mask = df["interest_rate"].between(lower, upper)
    n_iqr = int((~iqr_mask).sum())
    df = df[iqr_mask].copy()

    print(f"Removed target rows -> out-of-domain (<{t_lo} or >{t_hi}): {n_domain}, "
          f"IQR outliers [{lower:.2f}, {upper:.2f}]: {n_iqr}")
    print(f"Rows: {start_rows} -> {len(df)} "
          f"({start_rows - len(df)} dropped, all target-related)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4./5. FEATURE ENGINEERING + PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features, including a domain-motivated derived ratio."""
    df = df.copy()
    # Income-to-loan ratio: a higher ratio implies better repayment capacity.
    df["income_to_loan_ratio"] = df["annual_income"] / (df["loan_amount"] + 1)
    return df


NUMERIC_FEATURES = [
    "applicant_age",
    "years_employed",
    "credit_score",
    "annual_income",
    "loan_amount",
    "loan_term_months",
    "income_to_loan_ratio",
]
CATEGORICAL_FEATURES = ["loan_type"]
TARGET = "interest_rate"


def build_preprocessor() -> ColumnTransformer:
    """
    Impute + scale numerics, impute + one-hot encode categoricals.

    Doing this inside a ColumnTransformer (fit on train only) is the
    leakage-safe replacement for the original up-front median imputation
    and LabelEncoder usage. One-hot encoding is preferred over label
    encoding because loan_type is nominal (no inherent order).
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6./7. MODELLING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(name: str, model: Pipeline, X_test, y_test) -> dict:
    """Compute MAE / RMSE / R² for a fitted pipeline on the held-out set."""
    preds = model.predict(X_test)
    metrics = {
        "model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
    }
    print(
        f"{name:>22}  ->  MAE: {metrics['MAE']:.3f} | "
        f"RMSE: {metrics['RMSE']:.3f} | R2: {metrics['R2']:.4f}"
    )
    return metrics


def main() -> None:
    # 1. Load --------------------------------------------------------------
    df_raw = load_data()

    # 2. EDA ---------------------------------------------------------------
    run_eda(df_raw)

    # 3. Clean -------------------------------------------------------------
    df_clean = clean_data(df_raw)

    # 4./5. Features -------------------------------------------------------
    df_feat = add_features(df_clean)
    X = df_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    # FIX (Bug 3): reset the index on the modelling frame so feature/target
    # alignment is preserved through sklearn's internal fold shuffling. Building
    # X and y from the same reset-index frame guarantees they stay aligned.
    X = X.reset_index(drop=True)
    y = df_feat[TARGET].reset_index(drop=True)

    # 6. Split -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\n" + "=" * 70)
    print("4. MODEL TRAINING & EVALUATION")
    print("=" * 70)

    preprocessor = build_preprocessor()
    models = {

        "Baseline Mean": DummyRegressor(
            strategy="mean"
        ),

        "Ridge Regression": Ridge(
            alpha=1.0
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    results = []
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        results.append(evaluate(name, pipe, X_test, y_test))
        fitted[name] = pipe

    results_df = pd.DataFrame(results).set_index("model").round(4)
    results_df.to_csv(OUTPUT_DIR / "metrics.csv")

    # 8. Cross-validation on the best model --------------------------------
    print("\n" + "=" * 70)
    print("5. 5-FOLD CROSS-VALIDATION (best model)")
    print("=" * 70)
    best_name = results_df["R2"].idxmax()
    best_pipe = fitted[best_name]

    preds = best_pipe.predict(X_test)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_test,
        preds,
        alpha=0.6
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--"
    )

    plt.xlabel("Actual Interest Rate")
    plt.ylabel("Predicted Interest Rate")

    plt.title(
        f"Actual vs Predicted ({best_name})"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=120
    )

    plt.close()

    residuals = y_test - preds

    plt.figure(figsize=(7, 5))

    sns.histplot(
        residuals,
        bins=30,
        kde=True
    )

    plt.title(
        f"Residual Distribution ({best_name})"
    )

    plt.xlabel("Residual")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "residual_distribution.png",
        dpi=120
    )

    plt.close()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        best_pipe,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    mae_scores = -cross_val_score(
        best_pipe,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    print(f"Best model: {best_name}")
    print(f"CV R2 per fold: {np.round(cv_scores, 4)}")
    print(
        f"Mean CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
    )

    print(
        f"Mean CV MAE: {mae_scores.mean():.4f} +/- {mae_scores.std():.4f}"
    )

    # 9. Feature importance ------------------------------------------------
    print("\n" + "=" * 70)
    print("6. FEATURE IMPORTANCE (permutation, best model)")
    print("=" * 70)
    # Permutation importance works on the full pipeline and the original feature
    # columns, so it is robust to the one-hot expansion and easy to interpret.
    perm = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=20,
        random_state=RANDOM_STATE, scoring="r2", n_jobs=-1,
    )
    importances = (
        pd.Series(perm.importances_mean, index=X_test.columns)
        .sort_values(ascending=True)
    )
    print(importances.sort_values(ascending=False).round(4))

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="#4C72B0")
    plt.title(f"Permutation Feature Importance — {best_name}")
    plt.xlabel("Mean R² drop when feature is shuffled")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=120)
    plt.close()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Metrics table:\n{results_df}")
    print(f"\nAll figures + metrics.csv saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("BUSINESS INTERPRETATION")
    print("=" * 70)

    print(
        "\nApplicants likely to receive higher rates:"
    )

    print(
        "- Lower credit scores"
    )

    print(
        "- Lower income levels"
    )

    print(
        "- Certain loan categories such as Personal loans"
    )

    print(
        "\nAdditional data that would improve prediction:"
    )

    print(
        "- Debt-to-income ratio"
    )

    print(
        "- Existing liabilities"
    )

    print(
        "- Delinquency history"
    )

    print(
        "- Previous defaults"
    )

    print(
        "\nProduction monitoring:"
    )

    print(
        "- Data drift monitoring"
    )

    print(
        "- MAE tracking"
    )

    print(
        "- Quarterly retraining"
    )

if __name__ == "__main__":
    main()
