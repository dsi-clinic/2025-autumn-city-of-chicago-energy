"""stats_utils.py

Provides functions to generate descriptive statistics and missing-value summaries
for the Chicago Energy Benchmarking dataset.
"""

import numpy as np
import pandas as pd

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


def prepare_did_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for Difference-in-Differences (DiD) regression.

    Adds:
        - Post: 1 if Data Year >= POLICY_YEAR, else 0
        - LowRating: 1 if Chicago Energy Rating <= LOW_RATING_THRESHOLD, else 0
        - ln_FloorArea: log of Gross Floor Area
        - Interaction: Post × LowRating

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw Chicago Energy Benchmarking dataset.

    Returns:
    -------
    pd.DataFrame
        Prepared dataframe with treatment indicators, ready for DiD regression.
    """
    df_copy = dataframe.copy()

    # Convert to numeric types
    df_copy["Data Year"] = pd.to_numeric(df_copy["Data Year"], errors="coerce")
    df_copy["Chicago Energy Rating"] = pd.to_numeric(
        df_copy["Chicago Energy Rating"], errors="coerce"
    )
    df_copy["Gross Floor Area - Buildings (sq ft)"] = pd.to_numeric(
        df_copy["Gross Floor Area - Buildings (sq ft)"], errors="coerce"
    )

    # Treatment and time variables
    df_copy["Post"] = (df_copy["Data Year"] >= POLICY_YEAR).astype(int)
    df_copy["LowRating"] = (
        df_copy["Chicago Energy Rating"] <= LOW_RATING_THRESHOLD
    ).astype(int)

    # Log-transform floor area (avoid log(0))
    df_copy["ln_FloorArea"] = np.log(
        df_copy["Gross Floor Area - Buildings (sq ft)"].replace(0, np.nan)
    )

    # Interaction term
    df_copy["Interaction"] = df_copy["Post"] * df_copy["LowRating"]

    # Drop incomplete records
    df_copy = df_copy.dropna(
        subset=["Total GHG Emissions (Metric Tons CO2e)", "ln_FloorArea"]
    )

    print(f"[INFO] Prepared DiD dataset with {len(df_copy):,} observations.")
    print(f"[INFO] Share treated buildings: {df_copy['LowRating'].mean():.2%}")

    return df_copy


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
