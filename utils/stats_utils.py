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


def prepare_did_data(
    data: pd.DataFrame,
    *,
    mode: str = "chicago_low_rating",
    post_start_year: int | None = None,
    treated_city: str = "Chicago",
    outcome_col: str | None = None,
) -> pd.DataFrame:
    """Prepare dataset for Difference-in-Differences (DiD) analysis.

    Backward-compatible:
    - Default mode ("chicago_low_rating") matches your old notebooks:
        Post = 1{Data Year >= POLICY_YEAR}
        LowRating = 1{Chicago Energy Rating <= LOW_RATING_THRESHOLD}
        Interaction = Post * LowRating
        ln_FloorArea, ln_GHG are created.

    New mode:
    - mode="multicity" for pooled city comparison (Chicago vs other cities):
        Post = 1{Data Year >= post_start_year}  (recommended: 2020)
        LowRating = 1{City == treated_city}     (kept name for compatibility)
        Interaction = Post * LowRating
        ln_FloorArea created.
        If outcome_col is provided, it is coerced to numeric as well.

    Parameters
    ----------
    data:
        Input dataframe.
    mode:
        "chicago_low_rating" (default) or "multicity".
    post_start_year:
        Only used for mode="multicity". If None, defaults to 2020.
    treated_city:
        Only used for mode="multicity".
    outcome_col:
        Optional outcome column to coerce to numeric (useful for multicity Site EUI).

    Returns:
    -------
    pd.DataFrame
        Copy with DiD columns added and dtypes cleaned for statsmodels.
    """
    clean = data.copy()

    # ---- Common columns: coerce year and floor area when available ----
    if "Data Year" in clean.columns:
        clean["Data Year"] = pd.to_numeric(clean["Data Year"], errors="coerce")

    if "Gross Floor Area - Buildings (sq ft)" in clean.columns:
        clean["Gross Floor Area - Buildings (sq ft)"] = pd.to_numeric(
            clean["Gross Floor Area - Buildings (sq ft)"], errors="coerce"
        )

    if outcome_col is not None and outcome_col in clean.columns:
        clean[outcome_col] = pd.to_numeric(clean[outcome_col], errors="coerce")

    # ==================================================================
    # Mode 1 (default): Chicago-only treated = low Chicago Energy Rating
    # ==================================================================
    if mode == "chicago_low_rating":
        clean = clean.dropna(
            subset=[
                "Total GHG Emissions (Metric Tons CO2e)",
                "Gross Floor Area - Buildings (sq ft)",
                "Chicago Energy Rating",
                "Data Year",
            ]
        ).copy()

        clean["Post"] = (clean["Data Year"] >= POLICY_YEAR).astype("int64")
        clean["LowRating"] = (
            clean["Chicago Energy Rating"] <= LOW_RATING_THRESHOLD
        ).astype("int64")
        clean["Interaction"] = (clean["Post"] * clean["LowRating"]).astype("int64")

        clean["ln_FloorArea"] = np.log(clean["Gross Floor Area - Buildings (sq ft)"])
        clean["ln_GHG"] = np.log1p(clean["Total GHG Emissions (Metric Tons CO2e)"])

        # dtype hygiene for statsmodels/patsy
        clean["Data Year"] = clean["Data Year"].astype("int64")
        clean["Primary Property Type"] = clean["Primary Property Type"].astype("object")

        return clean

    # ==========================================================
    # Mode 2: Multi-city DiD (treated city vs other cities)
    # ==========================================================
    if mode == "multicity":
        if post_start_year is None:
            post_start_year = 2020

        clean = clean.dropna(
            subset=[
                "City",
                "Gross Floor Area - Buildings (sq ft)",
                "Data Year",
                "Primary Property Type",
            ]
            + ([outcome_col] if outcome_col is not None else [])
        ).copy()

        clean["Post"] = (clean["Data Year"] >= post_start_year).astype("int64")
        clean["LowRating"] = (clean["City"] == treated_city).astype("int64")
        clean["Interaction"] = (clean["Post"] * clean["LowRating"]).astype("int64")

        clean["ln_FloorArea"] = np.log(clean["Gross Floor Area - Buildings (sq ft)"])

        # dtype hygiene for statsmodels/patsy
        clean["Data Year"] = clean["Data Year"].astype("int64")
        clean["City"] = clean["City"].astype("object")
        clean["Primary Property Type"] = (
            clean["Primary Property Type"].astype(str).str.strip().str.lower()
        )

        if outcome_col is not None and outcome_col in clean.columns:
            clean[outcome_col] = clean[outcome_col].astype("float64")

        clean["ln_FloorArea"] = pd.to_numeric(
            clean["ln_FloorArea"], errors="coerce"
        ).astype("float64")

        clean = clean.dropna(
            subset=[
                "Post",
                "LowRating",
                "Interaction",
                "ln_FloorArea",
                "Data Year",
                "Primary Property Type",
            ]
            + ([outcome_col] if outcome_col is not None else [])
        ).copy()

        return clean

    raise ValueError(f"Unknown mode: {mode}")


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


def build_multi_city_did_df(
    chicago_df: pd.DataFrame,
    other_city_dfs: dict[str, pd.DataFrame],
    *,
    start_year: int = 2016,
    end_year: int = 2023,
    post_start_year: int = 2020,
    treated_city: str = "Chicago",
    city_col: str = "City",
    year_col: str = "Data Year",
    outcome_col: str = "Site EUI (kBtu/sq ft)",
    floor_area_col: str = "Gross Floor Area - Buildings (sq ft)",
    ptype_col: str = "Primary Property Type",
    chicago_city_value: str | None = "Chicago",
    ensure_city_values: bool = True,
    dropna_required: bool = True,
) -> pd.DataFrame:
    """Build a pooled multi-city DiD dataframe.

    Creates a single stacked dataset across cities and constructs:
      - Post: 1{year >= post_start_year}
      - LowRating: 1{City == treated_city}  (kept for backward-compatibility)
      - Interaction: Post * LowRating
      - ln_FloorArea: log(floor_area)

    Parameters
    ----------
    chicago_df:
        Chicago dataframe (may be concurrent_df or a cleaned variant).
    other_city_dfs:
        Dict of {city_name: df}. If ensure_city_values=True, the key is written to df[city_col]
        unless df already has city_col and chicago_city_value is None (see below).
    start_year, end_year:
        Year window filter applied after concatenation.
    post_start_year:
        "Post" period start; per your convention use 2020 to reflect 2019 reporting.
    treated_city:
        City value treated as "treated" for LowRating (default "Chicago").
    city_col, year_col, outcome_col, floor_area_col, ptype_col:
        Column names to use.
    chicago_city_value:
        If not None, writes this value into chicago_df[city_col] before concatenation.
        Set to None if chicago_df already has the correct City field and you don't want to overwrite it.
    ensure_city_values:
        If True, force-set each city's df[city_col] to the dict key (and set Chicago to chicago_city_value).
    dropna_required:
        If True, drop rows missing variables required for regression.

    Returns:
    -------
    pd.DataFrame
        Cleaned DiD-ready dataframe.
    """
    frames: list[pd.DataFrame] = []

    chi = chicago_df.copy()
    if ensure_city_values and chicago_city_value is not None:
        chi[city_col] = chicago_city_value
    frames.append(chi)

    for city_name, df in other_city_dfs.items():
        dfx = df.copy()
        if ensure_city_values:
            dfx[city_col] = city_name
        frames.append(dfx)

    did_df = pd.concat(frames, ignore_index=True)

    did_df[year_col] = pd.to_numeric(did_df[year_col], errors="coerce")
    did_df = did_df[did_df[year_col].between(start_year, end_year)].copy()

    did_df["Post"] = (did_df[year_col] >= post_start_year).astype(int)
    did_df["LowRating"] = (did_df[city_col] == treated_city).astype(int)
    did_df["Interaction"] = did_df["Post"] * did_df["LowRating"]

    floor_area = pd.to_numeric(did_df[floor_area_col], errors="coerce")
    did_df["ln_FloorArea"] = np.log(floor_area)

    did_df[outcome_col] = pd.to_numeric(did_df[outcome_col], errors="coerce")

    if dropna_required:
        did_df = did_df.dropna(
            subset=[
                outcome_col,
                "ln_FloorArea",
                ptype_col,
                year_col,
                city_col,
            ]
        ).copy()

    return did_df


def filter_property_type(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Filter dataframe to office or multifamily rows using Primary Property Type."""
    if "Primary Property Type" not in df.columns:
        raise KeyError("Expected column 'Primary Property Type' not found.")

    s = df["Primary Property Type"].astype(str).str.strip().str.lower()

    if kind == "office":
        mask = s.str.contains(r"\boffice\b", regex=True)

    elif kind == "multifamily":
        # catches: multifamily, multi-family, apartment, residential, housing, dorm etc.
        mask = s.str.contains(
            r"multi[-\s]?family|apartment|residential|housing|dorm|student housing",
            regex=True,
        )
    else:
        raise ValueError("kind must be 'office' or 'multifamily'")

    return df.loc[mask].copy()


def prep_for_did_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare df for level (no-log) DiD regression with statsmodels-safe dtypes."""
    out = df.copy()

    required = [
        "Site EUI (kBtu/sq ft)",
        "Post",
        "LowRating",
        "Interaction",
        "ln_FloorArea",
        "Data Year",
        "Primary Property Type",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns for DiD prep: {missing}")

    # outcome
    out["Site EUI (kBtu/sq ft)"] = pd.to_numeric(
        out["Site EUI (kBtu/sq ft)"], errors="coerce"
    ).astype("float64")

    # did vars as plain int64 (NOT pandas nullable Int64)
    for c in ["Post", "LowRating", "Interaction"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")

    # controls
    out["ln_FloorArea"] = pd.to_numeric(out["ln_FloorArea"], errors="coerce").astype(
        "float64"
    )

    # year as plain int64 (important for patsy)
    out["Data Year"] = pd.to_numeric(out["Data Year"], errors="coerce")
    out = out.dropna(subset=["Data Year"])
    out["Data Year"] = out["Data Year"].astype("int64")

    # categoricals
    out["Primary Property Type"] = (
        out["Primary Property Type"].astype(str).str.strip().str.lower()
    )
    if "City" in out.columns:
        out["City"] = out["City"].astype("object")

    # drop missing essentials
    out = out.dropna(
        subset=[
            "Site EUI (kBtu/sq ft)",
            "ln_FloorArea",
            "Primary Property Type",
            "Data Year",
        ]
    ).copy()

    return out
