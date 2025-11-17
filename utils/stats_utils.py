"""stats_utils.py

Provides functions to generate descriptive statistics and missing-value summaries
for the Chicago Energy Benchmarking dataset.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

logger = logging.getLogger(__name__)

PVAL_THRESHOLDS = [0.01, 0.05, 0.1]


CORE_COLS = [
    "Data Year",
    "Chicago Energy Rating",
    "Exempt From Chicago Energy Rating",
    "ENERGY STAR Score",
    "Site EUI (kBtu/sq ft)",
    "Source EUI (kBtu/sq ft)",
    "Weather Normalized Site EUI (kBtu/sq ft)",
    "Weather Normalized Source EUI (kBtu/sq ft)",
    "Total GHG Emissions (Metric Tons CO2e)",
    "GHG Intensity (kg CO2e/sq ft)",
    "Gross Floor Area - Buildings (sq ft)",
    "Year Built",
    "# of Buildings",
    "Primary Property Type",
    "Water Use (kGal)",
    "Electricity Use (kBtu)",
    "Natural Gas Use (kBtu)",
    "District Steam Use (kBtu)",
    "District Chilled Water Use (kBtu)",
    "All Other Fuel Use (kBtu)",
]

POLICY_YEAR = 2019
LOW_RATING_THRESHOLD = 2
PVAL_THRESHOLDS = (0.01, 0.05, 0.1)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def prepare_did_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset for Difference-in-Differences (DiD) analysis.

    Adds:
        - Post: 1 if Data Year >= 2019 (post-placard policy), else 0
        - LowRating: 1 if Chicago Energy Rating <= 2 (treated group), else 0
        - Interaction: Post * LowRating (the DiD term)
        - ln_FloorArea: natural log of Gross Floor Area
        - ln_GHG: log(1 + Total GHG Emissions)

    Parameters
    ----------
    df : pd.DataFrame
        Clean energy dataset with required columns.

    Returns:
    -------
    pd.DataFrame
        Copy with added DiD-related columns.
    """
    clean_data = data.copy()
    clean_data = clean_data.dropna(
        subset=[
            "Total GHG Emissions (Metric Tons CO2e)",
            "Gross Floor Area - Buildings (sq ft)",
            "Chicago Energy Rating",
            "Data Year",
        ]
    )

    clean_data["Post"] = (clean_data["Data Year"] >= POLICY_YEAR).astype(int)
    clean_data["LowRating"] = (
        clean_data["Chicago Energy Rating"] <= LOW_RATING_THRESHOLD
    ).astype(int)
    clean_data["Interaction"] = clean_data["Post"] * clean_data["LowRating"]
    clean_data["ln_FloorArea"] = np.log(
        clean_data["Gross Floor Area - Buildings (sq ft)"]
    )
    clean_data["ln_GHG"] = np.log1p(
        clean_data["Total GHG Emissions (Metric Tons CO2e)"]
    )

    return clean_data


# -----------------------------------------------------------------------------
# Difference-in-Differences regression
# -----------------------------------------------------------------------------


def run_did_regression(
    df: pd.DataFrame,
    y_var: str,
    log: bool = False,
    include_data_year: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run Difference-in-Differences OLS regression.

    Controls for building type and year fixed effects.
    """
    if log:
        y_var = "ln_GHG"

    formula = (
        f"Q('{y_var}') ~ Post + LowRating + Interaction + ln_FloorArea "
        "+ C(Q('Primary Property Type'))"
    )
    if include_data_year:
        formula += " + C(Q('Data Year'))"

    model = smf.ols(formula=formula, data=df).fit(cov_type="HC1")
    return model


# -----------------------------------------------------------------------------
# Summary utilities
# -----------------------------------------------------------------------------


def summarize_did_results(
    model: RegressionResultsWrapper,
    focus_terms: list[str] | None = None,
    highlight_energy_types: bool = True,
) -> pd.DataFrame:
    """Compact summary of DiD/DDD regression.

    Highlights energy-intensive property types (e.g., hospitals, gyms, labs).
    """
    table = model.summary2().tables[1].copy()

    p_col = next((c for c in ["P>|t|", "P>|z|", "P>|T|"] if c in table.columns), None)
    if p_col is None:
        raise KeyError("No valid p-value column found in regression output.")

    rename_map = {p_col: "p_value"}
    if "Coef." in table.columns:
        rename_map["Coef."] = "coef"
    if "Std.Err." in table.columns:
        rename_map["Std.Err."] = "std_err"

    table = table.rename(columns=rename_map)

    if focus_terms is None:
        focus_terms = ["Post", "LowRating", "Interaction", "ln_FloorArea"]

    mask = table.index.str.contains("|".join(focus_terms))

    if highlight_energy_types:
        energy_terms = [
            "hospital",
            "health club",
            "fitness center",
            "laboratory",
            "data center",
        ]
        mask |= table.index.str.contains("|".join(energy_terms), case=False)

    short = table[mask][["coef", "std_err", "p_value"]].round(4)

    short["Significance"] = short["p_value"].apply(
        lambda p: (
            "***"
            if p < PVAL_THRESHOLDS[0]
            else "**"
            if p < PVAL_THRESHOLDS[1]
            else "*"
            if p < PVAL_THRESHOLDS[2]
            else ""
        )
    )

    logger.info("Showing %d selected coefficients (policy + energy types).", len(short))
    return short


def summarize_missing_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return number and percentage of missing values per column, grouped by year.

    Parameters
    ----------
    df : pd.DataFrame

    Returns:
    -------
    pd.DataFrame
        Missing values summary by year.
    """
    year_col = "Data Year" if "Data Year" in df.columns else "Data_Year"
    grouped = []

    for year, group in df.groupby(year_col):
        total = len(group)
        missing = group.isna().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        missing["% Missing"] = (missing["Missing Values"] / total * 100).round(2)
        missing["Year"] = year
        grouped.append(missing)

    summary = pd.concat(grouped, ignore_index=True)
    return summary[["Year", "Column", "Missing Values", "% Missing"]].sort_values(
        ["Year", "% Missing"], ascending=[True, False]
    )


def generate_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the core analytical columns (not grouped).

    Parameters
    ----------
    df : pd.DataFrame

    Returns:
    -------
    pd.DataFrame
        Descriptive statistics for all years combined.
    """
    cols_present = [c for c in CORE_COLS if c in df.columns]
    return df[cols_present].describe(include="all").T


def generate_descriptive_stats_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for core analytical columns, grouped by year.

    Parameters
    ----------
    df : pd.DataFrame

    Returns:
    -------
    pd.DataFrame
        MultiIndex DataFrame with statistics per year and variable.
    """
    year_col = "Data Year" if "Data Year" in df.columns else "Data_Year"
    cols_present = [c for c in CORE_COLS if c in df.columns]

    grouped_stats = (
        df.groupby(year_col)[cols_present]
        .describe()
        .transpose()  # columns → rows for readability
    )

    # optional cleanup: rename index for readability
    grouped_stats.index.names = ["Variable", "Statistic"]

    return grouped_stats
