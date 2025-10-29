"""stats_utils.py"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ==============================================================
# Global constants
# ==============================================================

LOW_RATING_THRESHOLD: int = 2
ALPHA_STRONG: float = 0.001
ALPHA_MODERATE: float = 0.01
ALPHA_WEAK: float = 0.05

POLICY_YEAR: int = 2019
START_YEAR = 2016
END_YEAR = 2023

# ==============================================================
# Prepare data
# ==============================================================


def prepare_did_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for Difference-in-Differences regression.

    Adds:
    - Post (1 if Data Year >= POLICY_YEAR, else 0)
    - LowRating (1 if Chicago Energy Rating <= LOW_RATING_THRESHOLD, else 0)
    - ln_FloorArea = log(Gross Floor Area)
    - Interaction = Post × LowRating

    Parameters
    ----------
    df : pd.DataFrame
        Raw Chicago Energy Benchmarking dataset.

    Returns:
    -------
    pd.DataFrame
        Prepared dataframe with treatment indicators.
    """
    prepared_df = df.copy()

    prepared_df["Data Year"] = pd.to_numeric(prepared_df["Data Year"], errors="coerce")
    prepared_df["Chicago Energy Rating"] = pd.to_numeric(
        prepared_df["Chicago Energy Rating"], errors="coerce"
    )
    prepared_df["Gross Floor Area - Buildings (sq ft)"] = pd.to_numeric(
        prepared_df["Gross Floor Area - Buildings (sq ft)"], errors="coerce"
    )

    prepared_df["Post"] = (prepared_df["Data Year"] >= POLICY_YEAR).astype(int)
    prepared_df["LowRating"] = (
        prepared_df["Chicago Energy Rating"] <= LOW_RATING_THRESHOLD
    ).astype(int)
    prepared_df["ln_FloorArea"] = np.log(
        prepared_df["Gross Floor Area - Buildings (sq ft)"].replace(0, np.nan)
    )
    prepared_df["Interaction"] = prepared_df["Post"] * prepared_df["LowRating"]

    return prepared_df.dropna(
        subset=["Total GHG Emissions (Metric Tons CO2e)", "ln_FloorArea"]
    )


# ==============================================================
# DID Regression
# ==============================================================


def run_did_regression(
    df: pd.DataFrame,
    outcome_col: str,
    log: bool = False,
    safe_log: bool = False,
    include_data_year: bool = False,
    treatment_col: str = "LowRating",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run a Difference-in-Differences regression with optional log or safe-log.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    outcome_col : str
        Column name of the dependent variable.
    log : bool, default=False
        Apply log transform to outcome.
    safe_log : bool, default=False
        Use log(x + 1) transformation to retain nonpositive values.
    include_data_year : bool, default=False
        Add year fixed effects (C(Data Year)) if True.
    treatment_col : str, default="LowRating"
        Specify custom treatment indicator column.

    Returns:
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model with robust standard errors (HC3).
    """
    work_df = df.copy()

    if log:
        if safe_log:
            work_df[outcome_col] = np.log1p(work_df[outcome_col].clip(lower=0))
        else:
            invalid_rows = (work_df[outcome_col] <= 0).sum()
            if invalid_rows > 0:
                print(f"Dropping {invalid_rows} rows with nonpositive GHG values.")
                work_df = work_df[work_df[outcome_col] > 0].copy()
            work_df[outcome_col] = np.log(work_df[outcome_col])

    formula = (
        f"Q('{outcome_col}') ~ Post + {treatment_col} + Post:{treatment_col}"
        + " + np.log(Q('Gross Floor Area - Buildings (sq ft)'))"
        + " + Q('Year Built')"
    )
    if include_data_year:
        formula += " + C(Q('Data Year'))"

    model = smf.ols(formula=formula, data=work_df).fit(cov_type="HC3")

    return model


def summarize_did_results(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    show_all: bool = False,
) -> pd.DataFrame:
    """Summarize key DID regression outputs (coefficients, SEs, p-values).

    Parameters
    ----------
    model : statsmodels fitted model
        The regression result object.
    show_all : bool, default=False
        If True, display all coefficients (including Data Year dummies).
        If False, hide year dummies automatically.

    Returns:
    -------
    pd.DataFrame
        Clean summary table of coefficients and significance levels.
    """
    results = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "p_value": model.pvalues,
        }
    )

    def significance(p: float) -> str:
        if p < ALPHA_STRONG:
            return "***"
        if p < ALPHA_MODERATE:
            return "**"
        if p < ALPHA_WEAK:
            return "*"
        return ""

    results["Significance"] = results["p_value"].apply(significance)

    if not show_all:
        results = results[
            ~results.index.str.contains("C\\(Q\\('Data Year'\\)\\)", regex=True)
        ]

    return results.round(4)
