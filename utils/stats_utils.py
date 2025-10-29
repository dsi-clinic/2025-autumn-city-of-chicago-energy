"""stats_utils.py

Utilities for statistical analyses.

"""

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
