"""Utilities for loading and cleaning Chicago Energy Benchmarking data from CSV files."""

import json
import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from utils.settings import DATA_DIR

logger = logging.getLogger(__name__)
MIN_COMPLIANCE_YEAR = 2018
MIN_PRIMARY_PROPERTY = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def clean_numeric(series: pd.Series) -> pd.Series:
    """Cleaning columns to be numeric data type"""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan})
        .astype("float64", errors="ignore")
    )


def load_data() -> pd.DataFrame:
    """Load and clean Chicago Energy Benchmarking data from CSV files located in DATA_DIR."""
    path = DATA_DIR / "chicago_energy_benchmarking"

    # Backup absolute path for notebook use
    if not path.exists():
        path = Path("/project") / "data" / "chicago_energy_benchmarking"

    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    # Get all CSVs in this directory (non-recursive)
    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    # Load and concatenate all CSVs in parallel (2-4x faster initial load)
    with ThreadPoolExecutor() as executor:
        load_dfs = list(executor.map(pd.read_csv, csv_files))
    full_df = pd.concat(load_dfs, ignore_index=True)
    full_df = full_df.sort_values(by="Data Year")

    # Define columns
    str_cols = [
        "Property Name",
        "ZIP Code",
        "Community Area",
        "Primary Property Type",
        "Location",
        "Reporting Status",
        "Exempt From Chicago Energy Rating",
        "Row_ID",
    ]

    numeric_cols = [
        "Gross Floor Area - Buildings (sq ft)",
        "Electricity Use (kBtu)",
        "Natural Gas Use (kBtu)",
        "District Steam Use (kBtu)",
        "District Chilled Water Use (kBtu)",
        "All Other Fuel Use (kBtu)",
        "Site EUI (kBtu/sq ft)",
        "Source EUI (kBtu/sq ft)",
        "Weather Normalized Site EUI (kBtu/sq ft)",
        "Weather Normalized Source EUI (kBtu/sq ft)",
        "Total GHG Emissions (Metric Tons CO2e)",
        "GHG Intensity (kg CO2e/sq ft)",
        "Water Use (kGal)",
    ]

    # Convert string columns to lowercase
    full_df[str_cols] = full_df[str_cols].astype(str).apply(lambda col: col.str.lower())

    full_df = full_df.assign(
        **{
            col: clean_numeric(full_df[col])
            for col in numeric_cols
            if col in full_df.columns
        }
    )

    return full_df


def concurrent_buildings(
    input_df: pd.DataFrame = None,
    start_year: int = 2016,
    end_year: int = 2023,
    id_col: str = "ID",
    year_col: str = "Data Year",
    building_type_col: str = "Primary Property Type",
    building_type: list | None = None,
    status_col: str = "Reporting Status",
    submitted_label: str = "submitted",
    status_year: int = 2018,
) -> pd.DataFrame:
    """Filter buildings that have submitted data for all years in a specified range.

    Only records within [start_year, end_year] are kept. For years >= 2018,
    only rows whose reporting status matches one of the `submitted_labels`
    are considered.
    """
    if input_df is None:
        input_df = load_data()

    df_in_range = input_df[
        (input_df[year_col] >= start_year) & (input_df[year_col] <= end_year)
    ].copy()

    # Optional filter by building type
    if building_type:
        df_in_range = df_in_range[df_in_range[building_type_col].isin(building_type)]

    if status_col in df_in_range.columns:
        df_in_range[status_col] = df_in_range[status_col].str.strip().str.lower()
        df_in_range[status_col] = df_in_range[status_col].replace(
            "submitted data", "submitted"
        )

    # For years >= 2018, require Reporting Status in submitted_labels
    if status_col in df_in_range.columns:
        mask_pre_2018 = df_in_range[year_col] < status_year
        mask_2018_plus = df_in_range[year_col] >= status_year

        # keep all pre-2018 rows; filter 2018+ to submitted
        mask_submitted = df_in_range[status_col] == submitted_label
        df_in_range = df_in_range[mask_pre_2018 | (mask_2018_plus & mask_submitted)]

    required_years = set(range(start_year, end_year + 1))

    # Unique years per building
    building_years = (
        df_in_range.groupby(id_col)[year_col].unique().reset_index(name="Years")
    )

    # Buildings that have submitted in every required year
    buildings_all_years = building_years[
        building_years["Years"].apply(lambda years: required_years.issubset(set(years)))
    ]

    # Keep only those buildings, within year range
    filtered_df = df_in_range[
        df_in_range[id_col].isin(buildings_all_years[id_col])
    ].copy()

    # Ensure one row per building-year
    filtered_df = filtered_df.drop_duplicates(subset=[id_col, year_col], keep="first")

    return filtered_df


def pivot_energy_metric(
    metric_col: str,
    input_df: pd.DataFrame = None,
    start_year: int = 2016,
    end_year: int = 2023,
    id_col: str = "ID",
    year_col: str = "Data Year",
) -> pd.DataFrame:
    """Create a pivot table showing an energy metric over time for each building, and drop rows with missing values in the specified year range.

    Parameters
    ----------
    df : pd.DataFrame
        The energy dataset containing building and year info.
    metric_col : str
        The column name of the metric to pivot (e.g., 'Site EUI (kBtu/sq ft)').
    start_year : int, default = 2016
        The first year in the building range to consider for dropping nulls.
    end_year : int, default = 2023
        The last year in the building range to consider for dropping nulls.
    id_col : str, default="ID"
        Column identifying unique buildings.
    year_col : str, default="Data Year"
        Column indicating the reporting year.

    Returns:
    -------
    pd.DataFrame
        Pivoted DataFrame with buildings as rows and years as columns,
        containing the selected metric values. Rows with any null values
        in the specified year range are dropped.
    """
    if input_df is None:
        input_df = load_data()

    # Create pivot table
    pivot_df = input_df.pivot_table(index=id_col, columns=year_col, values=metric_col)

    # Identify the columns corresponding to the specified year range
    cols_to_check = [
        year for year in pivot_df.columns if start_year <= year <= end_year
    ]

    # Drop rows with any nulls in the specified year range
    pivot_df = pivot_df.dropna(subset=cols_to_check, how="any")

    # Optional metadata
    pivot_df.attrs["metric"] = metric_col
    pivot_df.attrs["num_buildings"] = pivot_df.shape[0]
    pivot_df.attrs["num_years"] = pivot_df.shape[1]
    pivot_df.attrs["year_range"] = (start_year, end_year)

    return pivot_df


def load_neighborhood_geojson() -> dict:
    """Loads the neighborhood GeoJSON file.

    Returns:
        A Python dictionary parsed from the GeoJSON file.
    """
    path = DATA_DIR / "chicago_geo"

    if not path.exists():
        path = Path("/project") / "data" / "chicago_geo"

    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    geojson_path = path / "neighborhood_chi.geojson"

    logger.info(f"Loading GeoJSON from: {geojson_path.resolve()}")
    with geojson_path.open() as f:
        geojson = json.load(f)

    logger.info(f"Loaded {len(geojson['features'])} features")
    return geojson


def load_community_geojson() -> dict:
    """Loads the community area GeoJSON file.

    Returns:
        A Python dictionary parsed from the GeoJSON file,
        with an added human-readable community_display field.
    """
    path = DATA_DIR / "chicago_geo"

    if not path.exists():
        path = Path("/project") / "data" / "chicago_geo"

    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    geojson_path = path / "Community_area_chi.geojson"

    logger.info(f"Loading GeoJSON from: {geojson_path.resolve()}")
    with geojson_path.open() as f:
        geojson = json.load(f)

    # ---- Add display-friendly community name ----
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        raw = props.get("community")

        if raw is not None and "community_display" not in props:
            props["community_display"] = " ".join(
                w.capitalize() for w in str(raw).lower().split()
            )

        feat["properties"] = props

    logger.info(f"Loaded {len(geojson['features'])} features")
    return geojson


def clean_property_type(energy_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure each building (ID) has a consistent Primary Property Type.

    Rules:
    1. If a building has only one valid type, fill all with that.
    2. Unify multifamily/residential variants → 'multifamily housing'.
    3. Unify senior care variants → 'senior care community'.
    4. Unify mall variants → 'mall'.
    5. Merge specified types → 'other'.
    """
    result_df = energy_df.copy()
    missing_vals = {"nan", "none", ""}

    # Your existing merge-to-other set
    merge_to_other = {
        "adult education",
        "other - education",
        "bank branch",
        "other - public services",
        "vehicle dealership",
        "courthouse",
        "financial office",
        "automobile dealership",
        "prison/incarceration",
        "pre-school/daycare",
        "repair services (vehicle, shoe, locksmith, etc.)",
        "lodging",
        "health care",
        "convention center",
        "outpatient rehabilitation/physical therapy",
        "commerce de détail",
        "urgent care/clinic/other outpatient",
        "other - services",
        "indoor arena",
    }

    # Step 1: Clean values and build per-building mapping (your existing logic)
    type_map = result_df.groupby("ID")["Primary Property Type"].apply(
        lambda x: [
            re.sub(r"\s+", " ", str(v)).strip().lower()
            for v in x
            if pd.notna(v) and str(v).strip().lower() not in missing_vals
        ]
    )

    id_to_type = {}
    for bid, types in type_map.items():
        lower_types = {t.strip().lower() for t in types}

        # Senior care, multifamily, mall, hospital, recreation logic (your existing)
        if lower_types & {"senior care community", "senior living community"}:
            id_to_type[bid] = "senior care community"
        elif len(lower_types) == 1:
            id_to_type[bid] = list(lower_types)[0]
        elif "multifamily housing" in lower_types:
            id_to_type[bid] = "multifamily housing"
        elif lower_types & {"enclosed mall", "strip mall", "other - mall"}:
            id_to_type[bid] = "mall"
        elif lower_types & {
            "hospital (general medical & surgical)",
            "other - specialty hospital",
        }:
            id_to_type[bid] = "hospital"
        elif "other - recreation" in lower_types:
            id_to_type[bid] = "recreation"
        elif lower_types & set(merge_to_other):
            id_to_type[bid] = "other"
        else:
            id_to_type[bid] = list(lower_types)[0] if lower_types else "other"

    # Step 2: Apply cleaned Primary Property Type (vectorized for 10-50x speedup)
    # First, normalize the Primary Property Type column
    normalized_types = (
        result_df["Primary Property Type"].astype(str).str.strip().str.lower()
    )
    is_missing = (
        normalized_types.isin(missing_vals) | result_df["Primary Property Type"].isna()
    )

    # Map IDs to types - use id_to_type for missing values, otherwise keep original
    result_df["Primary Property Type"] = (
        result_df["ID"]
        .map(id_to_type)
        .where(
            is_missing,
            result_df["ID"].map(id_to_type).fillna(result_df["Primary Property Type"]),
        )
    )

    # Step 3: NEW - Merge rare types (<150 instances) to "other"
    type_counts = result_df["Primary Property Type"].value_counts()
    rare_types = type_counts[type_counts < MIN_PRIMARY_PROPERTY].index.tolist()

    print(f"📊 Merging {len(rare_types)} rare types (<150) to 'other': {rare_types}")

    result_df["Primary Property Type"] = result_df["Primary Property Type"].replace(
        dict.fromkeys(rare_types, "other")
    )

    return result_df


def covid_impact_category(
    df: pd.DataFrame, property_col: str = "Primary Property Type", id_col: str = "ID"
) -> pd.DataFrame:
    """Assign each building a COVID impact category based on property type, without filtering by sample size.

    Categories:
        - Permanent: long-term reduction (remote work / downtown offices)
        - Temporary/Rebounded: short-term dip & later rebound
        - Stable/Increased: continuous or essential use
        - Other: uncertain or mixed-use categories that don't clearly fit
    """
    energy_df = df.copy()

    covid_mapping = {
        # --- Permanent reductions ---
        "office": "Permanent",
        "financial office": "Permanent",
        "bank branch": "Permanent",
        "commercial": "Permanent",
        # --- Temporary / Rebounded ---
        "k-12 school": "Temporary/Rebounded",
        "college/university": "Temporary/Rebounded",
        "hotel": "Temporary/Rebounded",
        "retail store": "Temporary/Rebounded",
        "supermarket/grocery store": "Temporary/Rebounded",
        "strip mall": "Temporary/Rebounded",
        "mall": "Temporary/Rebounded",
        "wholesale club/supercenter": "Temporary/Rebounded",
        "movie theater": "Temporary/Rebounded",
        "museum": "Temporary/Rebounded",
        "performing arts": "Temporary/Rebounded",
        "library": "Temporary/Rebounded",
        "fitness center/health club/gym": "Temporary/Rebounded",
        "indoor arena": "Temporary/Rebounded",
        "courthouse": "Temporary/Rebounded",
        "social/meeting hall": "Temporary/Rebounded",
        "lifestyle center": "Temporary/Rebounded",
        "convention center": "Temporary/Rebounded",
        "adult education": "Temporary/Rebounded",
        "pre-school/daycare": "Temporary/Rebounded",
        "residence hall/dormitory": "Temporary/Rebounded",
        "other - education": "Temporary/Rebounded",
        "other - recreation": "Temporary/Rebounded",
        "other - entertainment/public assembly": "Temporary/Rebounded",
        "other - lodging/residential": "Temporary/Rebounded",
        # --- Stable or Increased ---
        "multifamily housing": "Stable/Increased",
        "residential": "Stable/Increased",
        "senior care community": "Stable/Increased",
        "residential care facility": "Stable/Increased",
        "hospital (general medical & surgical)": "Stable/Increased",
        "other - specialty hospital": "Stable/Increased",
        "health care": "Stable/Increased",
        "medical office": "Stable/Increased",
        "urgent care/clinic/other outpatient": "Stable/Increased",
        "laboratory": "Stable/Increased",
        "worship facility": "Stable/Increased",
        "prison/incarceration": "Stable/Increased",
        "repair services (vehicle, shoe, locksmith, etc.)": "Stable/Increased",
        # --- Other (ambiguous or mixed) ---
        "mixed use property": "Other",
        "other": "Other",
        "not available": "Other",
        "other - services": "Other",
        "commerce de détail": "Other",
        "vehicle dealership": "Other",
        "automobile dealership": "Other",
        "outpatient rehabilitation/physical therapy": "Other",
        "medical office building": "Other",
        "lodging": "Other",
    }

    def categorize(prop: str | None) -> str:
        key = str(prop).strip().lower()
        return covid_mapping.get(key, "Other")  # default to "Other" if not in map

    energy_df["COVID Impact Category"] = energy_df[property_col].apply(categorize)

    logging.info("✅ COVID Impact Category assignment (with 'Other' group) complete.")
    logging.info(
        "Category counts:\n%s",
        energy_df["COVID Impact Category"].value_counts().sort_index().to_string(),
    )

    return energy_df


def assign_effective_year_built(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns the 'Effective Year Built' for each building ID.

    If one unique non-NaN year exists, it is assigned; if multiple years exist, assigns 'Multiple Years Built'; otherwise assigns np.nan.

    Args:
        df (pd.DataFrame): DataFrame with columns 'ID' and 'Year Built'.

    Returns:
        pd.DataFrame: Original DataFrame with new 'Effective Year Built' column.
    """

    def get_years(series: pd.Series) -> np.ndarray:
        unique_years = series.dropna().unique()
        if len(unique_years) == 1:
            # Building has one unique non-NaN value (regardless of number of NaNs)
            return np.repeat(unique_years[0], len(series))
        elif len(unique_years) > 1:
            # Building has multiple non-NaN values
            return np.repeat("Multiple Years Built", len(series))
        else:
            # Building has only NaNs
            return np.repeat(np.nan, len(series))

    df["Effective Year Built"] = df.groupby("ID")["Year Built"].transform(get_years)
    return df


def categorize_time_built(df: pd.DataFrame) -> pd.date_range:
    """Categorize buildings into construction period bins based on their 'Year Built'.

    Filters out entries where 'Effective Year Built' is missing or equals "Multiple Years Built".
    Then assigns a 'Decade Built' category to each remaining row based on the value in the 'Year Built' column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns 'Year Built' and 'Effective Year Built'.

    Returns:
    -------
    pd.DataFrame
        Filtered DataFrame with an added 'Decade Built' categorical column.
    """
    is_valid = (df["Effective Year Built"].notna()) & (
        df["Effective Year Built"] != "Multiple Years Built"
    )

    valid_df = df[is_valid].copy()
    bins = [0, 1920, 1960, 1990, 2010, float("inf")]
    labels = ["Before 1920", "1920-1960", "1960-1990", "1990-2010", "After 2010"]

    valid_df["Time Built"] = pd.cut(
        valid_df["Year Built"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    return valid_df


def prepare_persistence(
    df: pd.DataFrame,
    decade_built_col: str = "Time Built",
    site_eui_col: str = "Site EUI (kBtu/sq ft)",
) -> pd.DataFrame:
    """Prepare a DataFrame for energy persistence analysis by calculating year-to-year changes and aligning consecutive changes for comparison.

    The function filters and cleans input data, computes the year-over-year change in energy use (Delta)
    for each building, then aligns these changes to compare consecutive time intervals. The columns for
    construction period and site energy use are parameterized.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing building energy data.
    decade_built_col : str, optional
        Name of the column indicating the decade or period the building was constructed (default is 'Decade Built').
    site_eui_col : str, optional
        Name of the column with site energy use values (default is 'Site EUI (kBtu/sq ft)').

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing only valid rows, with columns for year-over-year energy change ('Delta')
        and the next year's change ('Delta_next') for each building.
    """
    cols = ["ID", "Data Year", decade_built_col, site_eui_col]
    site_df = df[cols].dropna().copy()

    site_df["Data Year"] = site_df["Data Year"].astype(int)
    site_df["ID"] = site_df["ID"].astype(str)
    site_df[decade_built_col] = site_df[decade_built_col].astype(str)
    site_df[site_eui_col] = pd.to_numeric(site_df[site_eui_col], errors="coerce")
    site_df = site_df.dropna(subset=[site_eui_col])

    df_delta = (
        site_df.sort_values(["ID", "Data Year"])
        .groupby("ID", group_keys=False)
        .apply(lambda g: g.assign(Delta=g[site_eui_col].diff()))
        .dropna(subset=["Delta"])
        .reset_index(drop=True)
    )

    df_lagged = (
        df_delta.sort_values([decade_built_col, "ID", "Data Year"])
        .groupby([decade_built_col, "ID"])
        .apply(lambda g: g.assign(Delta_next=g["Delta"].shift(-1)))
        .dropna(subset=["Delta", "Delta_next"])
        .reset_index(drop=True)
    )

    return df_lagged


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


def summarize_building(energy_df: pd.DataFrame, building_id: str | int) -> dict:
    """Summarize all relevant data for a given building ID.

    - String columns: show unique values horizontally across all years.
    - Numeric columns: show median values.
    - Excludes redundant metadata columns like ID, Data Year, and Location.

    Parameters
    ----------
    energy_df : pd.DataFrame
        The full dataset.
    building_id : str | int
        The building ID to summarize.

    Returns:
    -------
    dict
        A dictionary summary of all relevant building information.
    """
    if "ID" not in energy_df.columns:
        raise ValueError("The DataFrame must contain an 'ID' column.")

    building_data = energy_df[energy_df["ID"] == building_id]
    if building_data.empty:
        logger.warning(f"No records found for building ID {building_id}")
        return {}

    summary = {"Building ID": building_id}

    # Columns to skip
    skip_cols = {"ID", "Data Year", "Location", "Latitude", "Longitude", "Row_ID"}

    numeric_cols = [
        c
        for c in building_data.select_dtypes(include="number").columns
        if c not in skip_cols
    ]
    non_numeric_cols = [
        c
        for c in building_data.select_dtypes(exclude="number").columns
        if c not in skip_cols
    ]

    # Compute medians for numeric columns
    for col in numeric_cols:
        median_val = building_data[col].median(skipna=True)
        summary[col] = round(median_val, 2) if pd.notna(median_val) else None

    # Collect unique values for string columns
    for col in non_numeric_cols:
        unique_vals = sorted(
            {
                str(v).strip()
                for v in building_data[col].dropna().unique()
                if str(v).strip() != ""
            }
        )
        summary[col] = unique_vals

    # Log summary
    logger.info("=" * 100)
    logger.info(f"BUILDING SUMMARY — ID: {building_id}")
    logger.info("=" * 100)

    if "Data Year" in building_data.columns:
        years = building_data["Data Year"].dropna().unique()
        if len(years):
            logger.info(f"Years Recorded: {years.min()} → {years.max()}")
        logger.info("-" * 100)

    # Display non-numeric summaries horizontally
    for col in non_numeric_cols:
        vals = summary[col]
        if vals:
            if len(vals) == 1:
                logger.info(f"{col}: {vals[0]}")
            else:
                joined = "; ".join(vals)
                logger.info(f"{col}: {joined}")
    logger.info("-" * 100)

    # Display numeric summaries
    for col in numeric_cols:
        val = summary[col]
        logger.info(f"{col}: {val}")
    logger.info("=" * 100)

    return summary


# National Data
def load_national_eui_data() -> dict:
    """Return national reference statistics for Source EUI

    based on ENERGY STAR nationwide benchmarking data.
    """
    national_data = {
        "Data Year": [2018, 2019, 2020, 2021, 2022, 2023],
        "National_Mean_Source_EUI": [205.9, 202.1, 187.4, 186.5, 185.5, 180.5],
        "National_Median_Source_EUI": [142.1, 138.7, 124.1, 121.8, 118.1, 114.1],
    }
    return pd.DataFrame(national_data)


def load_covered_buildings() -> pd.DataFrame:
    """Load and clean the Chicago Energy Benchmarking *Covered Buildings* dataset.

    The dataset is loaded from CSV files located in
    DATA_DIR / 'chicago_energy_benchmarking_covered' and is expected
    to contain:
      - A unique Chicago Energy Benchmarking ID per property
      - Cohort / size information
      - Community Area, address, lat/long, etc.
    """
    path = DATA_DIR / "chicago_covered_buildings"

    # Backup absolute path for notebook use
    if not path.exists():
        path = Path("/project") / "data" / "chicago_covered_buildings"

    if not path.exists():
        raise FileNotFoundError(f"Covered Buildings data directory not found: {path}")

    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    # Load CSVs in parallel for faster performance
    with ThreadPoolExecutor() as executor:
        load_dfs = list(executor.map(pd.read_csv, csv_files))
    covered_df = pd.concat(load_dfs, ignore_index=True)

    # Normalize column names once here to match your benchmarking data
    col_renames = {
        "Building ID": "ID",
        "Verification Year": "Data Year",
    }

    covered_df = covered_df.rename(columns=col_renames)

    # Lowercase string-like columns for consistency
    for col in covered_df.select_dtypes(include="object").columns:
        covered_df[col] = covered_df[col].astype(str).str.lower().str.strip()

    # Ensure numeric types where relevant
    if "Data Year" in covered_df.columns:
        covered_df["Data Year"] = pd.to_numeric(
            covered_df["Data Year"], errors="coerce"
        )

    return covered_df


def find_out_of_compliance(
    start_year: int = 2016,
    end_year: int = 2023,
    id_col: str = "ID",
    year_col: str = "Data Year",
) -> pd.DataFrame:
    """Identify buildings assumed covered in a year but missing in reporting.

    Coverage is inferred by treating all buildings in the covered list
    as covered in every year from start_year through end_year, ignoring
    the verification year. Returns one row per (ID, year) where the
    building is covered but does not appear in the reporting data.
    """
    covered_expanded = expand_covered_buildings(
        start_year=start_year,
        end_year=end_year,
        id_col=id_col,
    )

    reported = load_data()

    # Normalize types
    covered_expanded[id_col] = covered_expanded[id_col].astype(str).str.strip()
    reported[id_col] = reported[id_col].astype(str).str.strip()

    covered_expanded[year_col] = pd.to_numeric(
        covered_expanded[year_col], errors="coerce"
    ).astype("Int64")
    reported[year_col] = pd.to_numeric(reported[year_col], errors="coerce").astype(
        "Int64"
    )

    covered_expanded = covered_expanded[
        (covered_expanded[year_col] >= start_year)
        & (covered_expanded[year_col] <= end_year)
    ].copy()
    reported = reported[
        (reported[year_col] >= start_year) & (reported[year_col] <= end_year)
    ].copy()

    # Unique (ID, year) pairs
    covered_pairs = covered_expanded[[id_col, year_col]].drop_duplicates()
    reported_pairs = reported[[id_col, year_col]].drop_duplicates()

    merged = covered_pairs.merge(
        reported_pairs,
        on=[id_col, year_col],
        how="left",
        indicator=True,
    )

    missing_pairs = merged[merged["_merge"] == "left_only"][[id_col, year_col]].rename(
        columns={year_col: "Missing Year"}
    )

    # Attach attributes from covered list
    attrs_cols = [
        id_col,
        year_col,
        "Cohort - Sector",
        "Cohort - Size",
        "Community Area Name",
        "Community Area Number",
        "Ward",
        "Latitude",
        "Longitude",
        "Location",
    ]
    attrs_cols = [c for c in attrs_cols if c in covered_expanded.columns]

    covered_attrs = covered_expanded[attrs_cols].drop_duplicates(
        subset=[id_col, year_col]
    )

    out_of_compliance = (
        missing_pairs.merge(
            covered_attrs,
            left_on=[id_col, "Missing Year"],
            right_on=[id_col, year_col],
            how="left",
        )
        .drop(columns=[year_col], errors="ignore")
        .drop_duplicates()
    )

    return out_of_compliance


def expand_covered_buildings(
    start_year: int,
    end_year: int,
    id_col: str = "ID",
) -> pd.DataFrame:
    """Treat every building in the covered list as covered in every year.

    Buildings are assumed covered from start_year through end_year, and
    mere presence in the covered list is interpreted as being subject
    to the ordinance.
    """
    covered = load_covered_buildings().copy()

    # Normalize ID
    covered[id_col] = covered[id_col].astype(str).str.strip()

    # Drop any duplicate IDs (keep first row as canonical attributes)
    covered_unique = covered.drop_duplicates(subset=[id_col]).copy()

    records = []
    for _, row in covered_unique.iterrows():
        for y in range(start_year, end_year + 1):
            r = row.to_dict()
            r["Data Year"] = y  # synthetic coverage year, not the verification year
            records.append(r)

    expanded = pd.DataFrame.from_records(records)

    return expanded


def covered_assign_top_types(
    covered_df: pd.DataFrame,
    source_col: str = "Cohort - Sector",
    target_col: str = "Top Level Property Type",
    other_label: str = "Other",
    title_case: bool = True,
) -> pd.DataFrame:
    """Create a standardized Top Level Property Type column for covered buildings.

    Parameters
    ----------
    covered_df : pd.DataFrame
        Covered buildings dataset.
    source_col : str
        Column containing sector/category information
        (default: "Cohort - Sector").
    target_col : str
        Name of the output column to create
        (default: "Top Level Property Type").
    other_label : str
        Label used for missing values.
    title_case : bool
        Whether to convert values to Title Case.

    Returns:
    -------
    pd.DataFrame
        Copy of covered_df with cleaned Top Level Property Type column added.
    """
    data = covered_df.copy()

    if source_col not in data.columns:
        raise KeyError(f"{source_col} not found in covered_df.")

    data[target_col] = data[source_col].fillna(other_label).astype(str).str.strip()

    if title_case:
        data[target_col] = data[target_col].str.title()

    data[target_col] = data[target_col].replace("", other_label)

    return data


def clean_year_built(
    energy_df: pd.DataFrame,
    id_col: str = "ID",
    year_col: str = "Data Year",
    year_built_col: str = "Year Built",
) -> pd.DataFrame:
    """Clean 'Year Built' for each building (ID), sorted by Data Year ascending."""
    cleaned_df = energy_df.copy()
    cleaned_df[year_built_col] = pd.to_numeric(
        cleaned_df[year_built_col], errors="coerce"
    )
    cleaned_df[year_col] = pd.to_numeric(cleaned_df[year_col], errors="coerce")

    def fix_building(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(year_col).reset_index(drop=True).copy()
        current: float | None = None
        cleaned_values: list[float | None] = [None] * len(group)

        # Forward pass: establish current year built
        for i, (_, row) in enumerate(group.iterrows()):
            year = row[year_col]
            year_built = row[year_built_col]

            if pd.notna(year_built) and pd.notna(year):
                if year_built == year:
                    # Rebuild year: update current
                    current = year_built
                elif current is None:
                    # First known value: establish current
                    current = year_built

            # Store forward-propagated value
            if current is not None:
                cleaned_values[i] = current

        # Backward pass: backfill current to earlier rows
        for i in range(len(group) - 1, -1, -1):
            if cleaned_values[i] is None and current is not None:
                cleaned_values[i] = current

        group[year_built_col] = cleaned_values
        return group

    result = cleaned_df.groupby(id_col, group_keys=False).apply(fix_building)

    # Reconstruct with original columns and preserve ID
    return result.reset_index(drop=True)


# --- compliance analysis ---


def filter_buildings_reported(
    energy_data: pd.DataFrame,
    energy_cols: list[str],
    reporting_status_col: str = "Reporting Status",
    allowed_statuses: list[str] | None = None,
    require_any_energy: bool = True,
) -> pd.DataFrame:
    """Filter energy benchmarking data to rows that count as 'reported'.

    A row is kept if:
      - It has at least one non-null energy/emissions metric (by default), AND
      - Its reporting status is in `allowed_statuses`

    Returns:
    -------
    pd.DataFrame
        Filtered dataframe of reported buildings.
    """
    if allowed_statuses is None:
        allowed_statuses = ["submitted", "submitted data", "nan"]

    missing_energy_cols = [c for c in energy_cols if c not in energy_data.columns]
    if missing_energy_cols:
        raise KeyError(f"Missing energy columns: {missing_energy_cols}")

    data = energy_data.copy()
    status_mask = data[reporting_status_col].isin(allowed_statuses)

    if require_any_energy:
        energy_mask = data[energy_cols].notna().any(axis=1)
    else:
        energy_mask = pd.Series(True, index=data.index)

    reported = data[status_mask & energy_mask].copy()
    return reported


def add_compliance_status(
    energy_data: pd.DataFrame,
    energy_cols: list[str],
    reporting_status_col: str = "Reporting Status",
    exempt_col: str = "Exempt From Chicago Energy Rating",
    allowed_statuses: list[str] | None = None,
    require_any_energy: bool = True,
    output_col: str = "compliance_status",
    inplace: bool = False,
) -> pd.DataFrame:
    """Normalize reporting status + add compliance labels

    - Label exempt-flag rows as 'exempt' first
    - If an exempt-flag row passes reported rule -> relabel to 'compliant'
    - For non-exempt rows:
        - reported -> 'compliant'
        - not reported -> 'non-compliant'

    Parameters
    ----------
    energy_data : pd.DataFrame
        Full energy benchmarking dataset.
    energy_cols : list[str]
        Columns used to determine whether a building has reported
        valid energy data.
    reporting_status_col : str, default "Reporting Status"
        Column containing raw reporting status values.
    exempt_col : str, default "Exempt From Chicago Energy Rating"
        Column indicating exemption status (True/False).
    allowed_statuses : list[str] | None
        Reporting statuses that count as "submitted".
    output_col : str
        Name of the new compliance status column.

    Returns:
    -------
    pd.DataFrame
        DataFrame with a new column `output_col`
        containing compliance labels.
    """
    data = energy_data if inplace else energy_data.copy()

    if reporting_status_col not in data.columns:
        raise KeyError(f"Missing column: {reporting_status_col}")

    data[reporting_status_col] = (
        data[reporting_status_col].astype(str).str.strip().str.lower()
    )
    data[reporting_status_col] = data[reporting_status_col].replace(
        {"submitted data": "submitted", "not covered 2024": "exempt"}
    )

    if allowed_statuses is None:
        allowed_statuses = ["submitted", "nan"]

    reported_df = filter_buildings_reported(
        energy_data=data,
        energy_cols=energy_cols,
        reporting_status_col=reporting_status_col,
        allowed_statuses=allowed_statuses,
        require_any_energy=require_any_energy,
    )
    reported_mask = data.index.isin(reported_df.index)

    if exempt_col in data.columns:
        exempt_true = data[exempt_col].astype(str).str.strip().str.lower().eq("true")
    else:
        exempt_true = pd.Series(False, index=data.index)

    data[output_col] = "other"  # default
    data.loc[exempt_true, output_col] = "exempt"
    data.loc[exempt_true & reported_mask, output_col] = "compliant"
    data.loc[(~exempt_true) & reported_mask, output_col] = "compliant"
    data.loc[(~exempt_true) & (~reported_mask), output_col] = "non-compliant"

    # NEW: force "not present in data" -> non-compliant
    data.loc[
        data[reporting_status_col].eq("not present in data"),
        output_col,
    ] = "non-compliant"

    return data


def build_compliance_base(
    energy_data: pd.DataFrame,
    year: int,
    year_col: str = "Data Year",
    id_col: str = "ID",
    area_col: str = "Community Area",
    status_col: str = "compliance_status",
    property_type_col: str = "Primary Property Type",
    top_level_col: str = "Top Level Property Type",
    time_built_col: str = "Time Built",
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a year-specific compliance base table.

    Filters the merged covered × benchmarking dataset to a single year and
    returns a simplified dataframe containing:

    - Building ID
    - Community area keys (area_key, area_display)
    - Classification columns (Primary Property Type, Top Level Property Type, Time Built)
    - Compliance status

    This base table is used to construct community-area summary tables for maps.

    Parameters
    ----------
    energy_data : pd.DataFrame
        Merged dataset containing compliance_status and classification fields.
    year : int
        Target reporting year.
    area_col : str
        Community area column.
    status_col : str
        Compliance status column.

    Returns:
    -------
    pd.DataFrame
        Filtered base dataframe for the specified year.
    """

    def norm_upper(x: str | None) -> str | None:
        return pd.NA if pd.isna(x) else str(x).strip().upper()

    if keep_cols is None:
        keep_cols = [property_type_col, top_level_col, time_built_col]
    keep_cols = [c for c in keep_cols if c in energy_data.columns]

    cols = [id_col, area_col, status_col] + keep_cols
    cols = list(dict.fromkeys(cols))

    base = energy_data.loc[energy_data[year_col] == year, cols].copy()

    base["_id"] = pd.to_numeric(base[id_col], errors="coerce")
    base["area_key"] = base[area_col].apply(norm_upper)
    base["area_display"] = base[area_col].astype("string").str.strip().str.title()
    base[status_col] = base[status_col].astype("string").str.strip().str.lower()

    # Clean grouping cols safely
    for c in keep_cols:
        base[c] = base[c].astype("string").str.strip()
        base[c] = base[c].where(base[c].notna(), pd.NA)

    if top_level_col in base.columns:
        base[top_level_col] = base[top_level_col].fillna("Other")
        base[top_level_col] = base[top_level_col].replace(
            {"nan": "Other", "NaN": "Other"}
        )

    base = base.dropna(subset=["_id", "area_key", status_col]).copy()
    base["_id"] = base["_id"].astype(int)

    dedup_cols = ["_id", "area_key"] + keep_cols
    base = base.drop_duplicates(subset=dedup_cols).copy()

    base = base.rename(columns={"_id": "Building ID"})
    out_cols = ["Building ID", "area_key", "area_display"] + keep_cols + [status_col]
    return base[out_cols]


def build_area_table_overall(
    base: pd.DataFrame,
    status_col: str = "compliance_status",
) -> pd.DataFrame:
    """Aggregate compliance statistics at the community-area level.

    Computes total counts and compliance shares for each community area
    from the year-specific base dataframe.

    Parameters
    ----------
    base : pd.DataFrame
        Year-filtered compliance base table produced by `build_compliance_base`.
    status_col : str
        Column indicating compliance status.

    Returns:
    -------
    pd.DataFrame
        Community-area summary table including:
        - area_key, area_display
        - num_submitted, num_non_compliant, denom
        - share_submitted, share_non_compliant
    """
    data = base.copy()
    data[status_col] = data[status_col].astype(str).str.strip().str.lower()

    counts = (
        data.groupby(["area_key", "area_display", status_col], as_index=False)
        .size()
        .pivot_table(
            index=["area_key", "area_display"],
            columns=status_col,
            values="size",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )

    for col in ["compliant", "non-compliant", "exempt"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.rename(
        columns={
            "compliant": "num_submitted",
            "non-compliant": "num_non_compliant",
        }
    )

    counts["denom"] = counts["num_submitted"] + counts["num_non_compliant"]
    counts["share_submitted"] = counts["num_submitted"] / counts["denom"].replace(
        0, pd.NA
    )
    counts["share_non_compliant"] = counts["num_non_compliant"] / counts[
        "denom"
    ].replace(0, pd.NA)

    return counts[
        [
            "area_key",
            "area_display",
            "num_submitted",
            "num_non_compliant",
            "exempt",
            "denom",
            "share_submitted",
            "share_non_compliant",
        ]
    ]


def build_area_table_by_property(
    base: pd.DataFrame,
    ptype_col: str,
    status_col: str = "compliance_status",
) -> pd.DataFrame:
    """Compute submitted/non-compliant counts + shares by area and a classification column."""
    data = base.copy()

    if ptype_col not in data.columns:
        raise KeyError(
            f"ptype_col='{ptype_col}' not found in base columns: {list(data.columns)}"
        )

    data[status_col] = data[status_col].astype(str).str.strip().str.lower()
    data = data[data[status_col].isin(["compliant", "non-compliant"])].copy()

    data[ptype_col] = data[ptype_col].where(data[ptype_col].notna(), pd.NA)
    data[ptype_col] = data[ptype_col].astype("string").str.strip()

    area_type = (
        data.groupby(
            ["area_key", "area_display", ptype_col, status_col], as_index=False
        )
        .size()
        .pivot_table(
            index=["area_key", "area_display", ptype_col],
            columns=status_col,
            values="size",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )

    for col in ["compliant", "non-compliant"]:
        if col not in area_type.columns:
            area_type[col] = 0

    area_type = area_type.rename(
        columns={
            ptype_col: "ptype_key",
            "compliant": "num_submitted",
            "non-compliant": "num_non_compliant",
        }
    )

    area_type["denom"] = area_type["num_submitted"] + area_type["num_non_compliant"]
    area_type["share_non_compliant"] = area_type["num_non_compliant"] / area_type[
        "denom"
    ].replace(0, pd.NA)

    area_type["share_submitted"] = area_type["num_submitted"] / area_type[
        "denom"
    ].replace(0, pd.NA)

    area_type["_lookup_key"] = (
        area_type["area_key"] + "|" + area_type["ptype_key"].astype("string")
    )

    return area_type[
        [
            "area_key",
            "area_display",
            "ptype_key",
            "num_submitted",
            "num_non_compliant",
            "denom",
            "share_submitted",
            "share_non_compliant",
            "_lookup_key",
        ]
    ]


def compliance_by_category(
    energy_data: pd.DataFrame,
    year: int,
    category_col: str,
    year_col: str = "Data Year",
    id_col: str = "ID",
    status_col: str = "compliance_status",
) -> pd.DataFrame:
    """Compute compliance and non‑compliance counts and rates by category for a given year.

    Parameters
    ----------
    energy_data : pd.DataFrame
        Full dataset of buildings, including both reporting and non‑reporting buildings
        for all years.
    energy_reported : pd.DataFrame
        Subset of buildings that successfully reported (i.e., are considered compliant),
        typically filtered to rows with valid reporting status.
    year : int
        Single calendar year (from the `year_col`) for which compliance should be
        calculated.
    category_col : str
        Column name in `energy_data` used to group buildings (e.g., "Primary Property Type",
        "Top Level Property Type", "Community Area").
    year_col : str, optional
        Name of the column containing the data year, by default "Data Year".
    id_col : str, optional
        Column name representing a unique building identifier, by default "ID".

    Returns:
    -------
    pd.DataFrame
        Summary table with one row per unique value in `category_col` for the specified
        year, including:
        - `category_col`: the category value (group).
        - `compliant`: count of compliant buildings in that category.
        - `non_compliant`: count of non‑compliant buildings in that category.
        - `total`: total number of buildings in that category (`compliant + non_compliant`).
        - `non_compliance_rate`: fraction of non‑compliant buildings (`non_compliant / total`),
          sorted in descending order of `total`.
    """
    if status_col not in energy_data.columns:
        raise KeyError(f"'{status_col}' not found. Run add_compliance_status() first.")

    data = energy_data[energy_data[year_col] == year].copy()
    data[status_col] = data[status_col].astype(str).str.strip().str.lower()
    data = data[data[status_col].isin(["compliant", "non-compliant"])].copy()

    summary = data.pivot_table(
        index=category_col,
        columns=status_col,
        values=id_col,
        aggfunc="count",
        fill_value=0,
    ).reset_index()

    if "compliant" not in summary.columns:
        summary["compliant"] = 0
    if "non-compliant" not in summary.columns:
        summary["non-compliant"] = 0

    summary = summary.rename(columns={"non-compliant": "non_compliant"})

    summary["total"] = summary["compliant"] + summary["non_compliant"]
    summary["non_compliance_rate"] = summary["non_compliant"] / summary[
        "total"
    ].replace(0, pd.NA)

    return summary.sort_values("total", ascending=False)


def merge_covered_with_benchmarking(
    covered_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    data_year_col: str = "Data Year",
    status_col: str = "Reporting Status",
    missing_label: str = "Not present in data",
    id_col: str = "ID",
) -> pd.DataFrame:
    """Add synthetic rows for covered buildings that do not appear in the benchmarking data for each year.

    Parameters
    ----------
    covered_df : pd.DataFrame
        Covered buildings dataset containing all buildings that should benchmark
        (one row per covered building, with at least `Building ID`, address, ZIP,
        community area, and location metadata).
    benchmark_df : pd.DataFrame
        Benchmarking dataset containing buildings that actually reported for one
        or more years (may have multiple rows per building across years).
    data_year_col : str, optional
        Name of the column in `benchmark_df` and the output that stores the data year,
        by default "Data Year".
    status_col : str, optional
        Name of the reporting status column in `benchmark_df` and the output,
        by default "Reporting Status".
    missing_label : str, optional
        Status label to assign to synthetic rows for covered buildings that do not
        appear in the benchmarking data for a given year, by default
        "Not present in data".
    id_col : str, optional
        Column name to use as the standardized building identifier in both dataframes.
        In `covered_df` this is expected as "Building ID" and will be renamed;
        in `benchmark_df` this is expected as "ID" and will be renamed, by default "ID".

    Returns:
    -------
    pd.DataFrame
        Combined benchmarking dataframe with:
        - All original rows from `benchmark_df`, unchanged.
        - Additional rows for each year and each covered building that is absent
          from the benchmarking data in that year, with:
            * `Data Year` set to the corresponding year.
            * `Reporting Status` set to `missing_label`.
            * ID and key location columns (Address, ZIP Code, Community Area,
              Latitude, Longitude, Location) populated from `covered_df` when available.
            * All other benchmarking columns filled with NaN.
    """
    # Standardize IDs
    covered_df = covered_df.rename(columns={"Building ID": id_col}).copy()
    benchmark_df = benchmark_df.rename(columns={"ID": id_col}).copy()

    bench_cols = benchmark_df.columns.tolist()
    map_cols = {
        "Address": "Address",
        "Zip": "ZIP Code",
        "Community Area Name": "Community Area",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "Location": "Location",
        "Top Level Property Type": "Top Level Property Type",
    }

    years = sorted(benchmark_df[data_year_col].drop_duplicates())
    result = benchmark_df.copy()
    total_added = 0

    for year in years:
        # Benchmarking IDs this year ONLY
        year_mask = (result[data_year_col] == year) & result[id_col].notna()
        year_ids = set(result.loc[year_mask, id_col].unique())

        # ALL covered buildings absent this year (NO drop_duplicates)
        absent_covered = covered_df[~covered_df[id_col].isin(year_ids)].copy()

        num_new = len(absent_covered)
        if num_new == 0:
            continue

        # Create rows matching covered shape exactly
        new_rows = pd.DataFrame(np.nan, index=absent_covered.index, columns=bench_cols)
        new_rows[data_year_col] = year
        new_rows[status_col] = missing_label
        new_rows[id_col] = absent_covered[id_col]

        # Map ALL available columns
        for cov_col, bench_col in map_cols.items():
            if cov_col in covered_df.columns:
                new_rows[bench_col] = absent_covered[cov_col]

        result = pd.concat([result, new_rows], ignore_index=True)
        total_added += num_new
        print(f"Year {year}: added {num_new} rows")

    print(f"✅ Total added: {total_added} rows across all years")
    return result


def merge_covered_with_benchmarking_new(
    covered_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    data_year_col: str = "Data Year",
    status_col: str = "Reporting Status",
    missing_label: str = "Not present in data",
    id_col: str = "ID",
    primary_ptype_col: str = "Primary Property Type",
    top_level_col: str = "Top Level Property Type",
    year_built_col: str = "Year Built",
    effective_year_built_col: str = "Effective Year Built",
    verbose: bool = True,
) -> pd.DataFrame:
    """Add synthetic rows for covered buildings absent from benchmarking per year.

    Synthetic-row rules:
    - Primary Property Type:
        * If ID appears in benchmarking at least once: use that ID’s most common primary property type.
        * If ID never appears: use top-level fallback (mode primary type within that top-level in benchmarking).
    - Effective Year Built:
        * If ID appears in benchmarking at least once: use that ID’s Effective Year Built (from benchmarking).
        * If ID never appears: leave Effective Year Built = np.nan (as requested).
    """
    covered_df = covered_df.rename(columns={"Building ID": id_col}).copy()
    benchmark_df = benchmark_df.rename(columns={"ID": id_col}).copy()

    for col in [primary_ptype_col, top_level_col]:
        if col not in benchmark_df.columns:
            raise KeyError(f"benchmark_df missing required column: '{col}'")

    if primary_ptype_col not in covered_df.columns:
        covered_df[primary_ptype_col] = pd.NA
    if top_level_col not in covered_df.columns:
        covered_df[top_level_col] = pd.NA

    bench_cols = benchmark_df.columns.tolist()
    for needed in [
        primary_ptype_col,
        top_level_col,
        year_built_col,
        effective_year_built_col,
    ]:
        if needed not in bench_cols:
            bench_cols.append(needed)

    map_cols = {
        "Address": "Address",
        "Zip": "ZIP Code",
        "Community Area Name": "Community Area",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "Location": "Location",
        top_level_col: top_level_col,
    }

    bench_id_primary = (
        benchmark_df.dropna(subset=[id_col, primary_ptype_col])
        .groupby(id_col)[primary_ptype_col]
        .agg(lambda s: s.value_counts(dropna=True).index[0])
    )
    bench_ids = set(bench_id_primary.index)

    # top-level -> mode(primary)
    top_level_to_mode_primary = (
        benchmark_df.dropna(subset=[top_level_col, primary_ptype_col])
        .groupby(top_level_col)[primary_ptype_col]
        .agg(lambda s: s.value_counts(dropna=True).index[0])
        .to_dict()
    )

    covered_ids = set(covered_df[id_col].dropna().unique())
    never_ids = covered_ids - bench_ids

    id_to_primary: dict = {}
    id_to_primary.update(bench_id_primary.to_dict())

    if len(never_ids) > 0:
        never_meta = covered_df.loc[
            covered_df[id_col].isin(list(never_ids)), [id_col, top_level_col]
        ].copy()
        never_meta[top_level_col] = never_meta[top_level_col].astype(str).str.strip()

        for _id, tl in zip(never_meta[id_col], never_meta[top_level_col]):
            id_to_primary[_id] = top_level_to_mode_primary.get(tl, pd.NA)

    # NOTE: We only assign this for IDs that appear in benchmarking.
    bench_years = benchmark_df[[id_col, year_built_col]].copy()
    bench_years[year_built_col] = pd.to_numeric(
        bench_years[year_built_col], errors="coerce"
    )

    def _effective(series: pd.Series) -> pd.Series:
        uniq = series.dropna().unique()
        if len(uniq) == 1:
            return pd.Series([uniq[0]] * len(series), index=series.index)
        elif len(uniq) > 1:
            return pd.Series(["Multiple Years Built"] * len(series), index=series.index)
        return pd.Series([np.nan] * len(series), index=series.index)

    id_to_effective_yb = (
        bench_years.groupby(id_col)[year_built_col].apply(_effective).to_dict()
    )
    # For never_ids, we deliberately do NOT set anything (leave as NaN)

    if verbose:
        print(f"Covered IDs: {len(covered_ids):,}")
        print(f"Benchmark IDs (ever): {len(bench_ids):,}")
        print(f"Covered never in benchmark: {len(never_ids):,}")

    # add synthetic rows per year
    years = sorted(benchmark_df[data_year_col].drop_duplicates())
    result = benchmark_df.copy()
    total_added = 0

    for year in years:
        year_mask = (benchmark_df[data_year_col] == year) & benchmark_df[id_col].notna()
        year_ids = set(benchmark_df.loc[year_mask, id_col].unique())

        absent_covered = covered_df.loc[~covered_df[id_col].isin(year_ids)].copy()
        num_new = len(absent_covered)
        if num_new == 0:
            continue

        new_rows = pd.DataFrame(np.nan, index=absent_covered.index, columns=bench_cols)
        new_rows[data_year_col] = year
        new_rows[status_col] = missing_label
        new_rows[id_col] = absent_covered[id_col]

        # Top level from covered (if present)
        if top_level_col in absent_covered.columns:
            new_rows[top_level_col] = absent_covered[top_level_col]

        # Primary property type assignment
        new_rows[primary_ptype_col] = new_rows[id_col].map(id_to_primary)

        # Effective Year Built assignment: only for IDs that appear in benchmarking
        new_rows[effective_year_built_col] = new_rows[id_col].map(id_to_effective_yb)

        for cov_col, bench_col in map_cols.items():
            if cov_col in absent_covered.columns and bench_col in new_rows.columns:
                new_rows[bench_col] = absent_covered[cov_col]

        result = pd.concat([result, new_rows], ignore_index=True)
        total_added += num_new
        if verbose:
            print(f"Year {year}: added {num_new} rows")

    if verbose:
        print(f"✅ Total added: {total_added} rows across all years")

    return result


def add_reporting_compliance_flags(
    benchmark_df: pd.DataFrame,
    year_col: str = "Data Year",
    status_col: str = "Reporting Status",
) -> pd.DataFrame:
    """Standardize reporting status and add compliance flags for 2018+ records.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Full benchmarking dataset containing at least a year column and a reporting
        status column. May include records from years before and after 2018.
    year_col : str, optional
        Name of the column indicating the data year for each record, by default
        "Data Year".
    status_col : str, optional
        Name of the column containing the original reporting status text
        (e.g., "Submitted", "Not Submitted", "Exempt", "Submitted Data"),
        by default "Reporting Status".

    Returns:
    -------
    pd.DataFrame
        Copy of the input dataframe filtered to `MIN_COMPLIANCE_YEAR` and later,
        with a cleaned/normalized reporting status column and the following
        boolean flag columns added:

        - `SubmittedFlag`: True if cleaned status equals "submitted".
        - `ExemptFlag`: True if cleaned status equals "exempt".
        - `NotSubmittedFlag`: True if cleaned status equals "not submitted".
        - `NonCompliantFlag`: True if status is "not submitted" and not exempt
          (i.e., counted as non‑compliant for ordinance purposes).
    """
    compliance_df = benchmark_df.copy()

    # Only keep MIN_COMPLIANCE_YEAR+ for these analyses
    compliance_df = compliance_df[compliance_df[year_col] >= MIN_COMPLIANCE_YEAR].copy()

    # Clean and normalize status
    compliance_df[status_col] = (
        compliance_df[status_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("submitted data", "submitted")
    )

    # Basic flags
    compliance_df["SubmittedFlag"] = compliance_df[status_col].eq("submitted")
    compliance_df["ExemptFlag"] = compliance_df[status_col].eq("exempt")
    compliance_df["NotSubmittedFlag"] = compliance_df[status_col].eq("not submitted")

    # Overall non‑compliance flag: not submitted and not exempt
    compliance_df["NonCompliantFlag"] = (
        compliance_df["NotSubmittedFlag"] & ~compliance_df["ExemptFlag"]
    )

    return compliance_df


def add_top_level_property_type(
    benchmark_df: pd.DataFrame,
    source_col: str = "Primary Property Type",
    target_col: str = "Top Level Property Type",
) -> pd.DataFrame:
    """Create a 4‑bucket top‑level property type column from detailed property types.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        DataFrame containing building‑level records with a detailed property type
        column (e.g., cleaned Portfolio Manager `Primary Property Type`).
    source_col : str, optional
        Name of the column in `benchmark_df` that holds the detailed property type
        values to be grouped, by default "Primary Property Type".
    target_col : str, optional
        Name of the new column to be added to `benchmark_df` containing the
        top‑level classification ("Residential", "Commercial", "Municipal", or "Other"),
        by default "Top Level Property Type".

    Returns:
    -------
    pd.DataFrame
        Copy of `benchmark_df` with an additional `target_col` where each row's
        detailed property type is mapped into one of four buckets:

        - "Residential": multifamily housing, residential, hotel, senior care community,
          residence hall/dormitory, mixed use property.
        - "Commercial": office, retail store, commercial, supermarket/grocery store,
          mall, strip mall, medical office, laboratory, hospital (general medical & surgical).
        - "Municipal": k‑12 school, college/university.
        - "Other": any type not explicitly listed in the mapping or missing.
    """
    mapping = {
        # Residential
        "multifamily housing": "Residential",
        "residential": "Residential",
        "hotel": "Residential",
        "senior care community": "Residential",
        "residence hall/dormitory": "Residential",
        "mixed use property": "Residential",
        # Commercial
        "office": "Commercial",
        "retail store": "Commercial",
        "commercial": "Commercial",
        "supermarket/grocery store": "Commercial",
        "mall": "Commercial",
        "strip mall": "Commercial",
        "medical office": "Commercial",
        "laboratory": "Commercial",
        "hospital (general medical & surgical)": "Commercial",
        # Municipal
        "k-12 school": "Municipal",
        "college/university": "Municipal",
        # Other (already "other")
        "other": "Other",
    }

    result_df = benchmark_df.copy()

    def classify_top_level(raw_type: object) -> str:
        if pd.isna(raw_type):
            return "Other"
        key = str(raw_type).strip().lower()
        return mapping.get(key, "Other")

    result_df[target_col] = result_df[source_col].apply(classify_top_level)

    return result_df


def load_major_us_cities() -> dict[str, pd.DataFrame]:
    """Load all major US city datasets.

    - Top-level CSV files are loaded directly.
    - Boston_data folder is loaded using load_boston_energy_data().
    - Returns: {key: DataFrame}
    Keys:
        - For top-level CSV files: the file name stem
          (e.g., "Seattle_Benchmarking_Performance_Ranges_by_Building_Type").
        - For the Boston_data folder: "boston_energy_raw".
    """
    path = DATA_DIR / "major_us_cities_data"

    # Backup path for /project environments
    if not path.exists():
        path = Path("/project") / "data" / "major_us_cities_data"

    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    city_data: dict[str, pd.DataFrame] = {}

    csv_files = sorted(path.glob("*.csv"))

    # Load city CSVs in parallel for faster performance
    def load_city_file(file: Path) -> tuple[str, pd.DataFrame]:
        """Load a single city CSV file."""
        df_city = pd.read_csv(file, low_memory=False)
        logger.info("Loaded %s → %s", file.name, df_city.shape)
        return file.stem, df_city

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_city_file, csv_files))

    for key, df_city in results:
        city_data[key] = df_city

    boston_folder = path / "Boston_data"
    if boston_folder.exists():
        df_boston = load_boston_energy_data(boston_folder)
        city_data["boston_energy_raw"] = df_boston
        logger.info(
            "Loaded Boston_data folder → %s",
            df_boston.shape,
        )

    if not city_data:
        raise FileNotFoundError(f"No datasets found in {path}")

    return city_data


# -----------------------------------------------------------------------------------
# ---------------------------- Merge Major City Data --------------------------------
# -----------------------------------------------------------------------------------

CHICAGO_CANONICAL_COLS = {
    "Data Year",
    "ID",
    "Property Name",
    "Address",
    "ZIP Code",
    "Primary Property Type",
    "Gross Floor Area - Buildings (sq ft)",
    "Site EUI (kBtu/sq ft)",
    "Source EUI (kBtu/sq ft)",
    "ENERGY STAR Score",
}


@dataclass(frozen=True)
class CitySchema:
    """Defines how to map a city's raw columns into Chicago's canonical schema."""

    city: str
    column_map: dict[str, str]
    required_cols: Iterable[str] = ("Data Year", "Address", "Primary Property Type")


def _standardize_strings(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (
                out[c]
                .astype(str)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "none": np.nan})
            )
    return out


def _apply_chicago_style_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a light version of Chicago-style cleaning for city data.

    IMPORTANT: avoid lowercasing Address if you merge on Address later.
    """
    out = df.copy()

    numeric_cols = [
        "Gross Floor Area - Buildings (sq ft)",
        "Site EUI (kBtu/sq ft)",
        "Source EUI (kBtu/sq ft)",
        "ENERGY STAR Score",
    ]
    out = out.assign(
        **{col: clean_numeric(out[col]) for col in numeric_cols if col in out.columns}
    )

    str_cols = ["Property Name", "Primary Property Type", "ZIP Code"]
    out = _standardize_strings(out, str_cols)

    return out


# --- Chicago schema (exact) ---
CHICAGO_COLS = [
    "Data Year",
    "ID",
    "Property Name",
    "Address",
    "ZIP Code",
    "Community Area",
    "Primary Property Type",
    "Gross Floor Area - Buildings (sq ft)",
    "Year Built",
    "# of Buildings",
    "ENERGY STAR Score",
    "Electricity Use (kBtu)",
    "Natural Gas Use (kBtu)",
    "District Steam Use (kBtu)",
    "District Chilled Water Use (kBtu)",
    "All Other Fuel Use (kBtu)",
    "Site EUI (kBtu/sq ft)",
    "Source EUI (kBtu/sq ft)",
    "Weather Normalized Site EUI (kBtu/sq ft)",
    "Weather Normalized Source EUI (kBtu/sq ft)",
    "Total GHG Emissions (Metric Tons CO2e)",
    "GHG Intensity (kg CO2e/sq ft)",
    "Latitude",
    "Longitude",
    "Location",
    "Reporting Status",
    "Chicago Energy Rating",
    "Exempt From Chicago Energy Rating",
    "Water Use (kGal)",
    "Row_ID",
]

KWH_TO_KBTU = 3.412141633


def _to_num(s: pd.Series | None) -> pd.Series:
    """Coerce a Series-like object to numeric; returns float with NaNs for bad values."""
    if s is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(s, errors="coerce")


def _make_location(
    df_city: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.Series:
    """Create Chicago-style Location string: 'POINT (lon lat)' where lat/lon present."""
    lat = _to_num(df_city.get(lat_col))
    lon = _to_num(df_city.get(lon_col))

    loc = pd.Series(pd.NA, index=df_city.index, dtype="object")
    ok = lat.notna() & lon.notna()
    loc.loc[ok] = (
        "POINT (" + lon.loc[ok].astype(str) + " " + lat.loc[ok].astype(str) + ")"
    )
    return loc


def sf_to_chicago(sf_df: pd.DataFrame) -> pd.DataFrame:
    """Map San Francisco benchmarking data into Chicago's schema.

    Assumes SF columns (common from SF Open Data export) like:
    - Benchmark Year, unique_identifier, Building Name, Building Address, Postal Code
    - Category, Floor Area, Year Built, ENERGY STAR Score
    - Electricity Use - Grid Purchase (kWh), Natural Gas Use (kBtu), District Steam Use (kBtu)
    - Site EUI (kBtu/ft2), Source EUI (kBtu/ft2), Weather Normalized ... (kBtu/ft2)
    - Total GHG Emissions ..., Total GHG Emissions Intensity ...
    - latitude, longitude, Benchmark Status, Reason for Exemption
    """
    out = pd.DataFrame(index=sf_df.index)

    out["Data Year"] = _to_num(sf_df.get("Benchmark Year"))
    out["ID"] = sf_df.get("unique_identifier")

    out["Property Name"] = sf_df.get("Building Name")
    out["Address"] = sf_df.get("Building Address")
    out["ZIP Code"] = sf_df.get("Postal Code")

    out["Community Area"] = pd.NA
    out["Primary Property Type"] = sf_df.get("Category")

    out["Gross Floor Area - Buildings (sq ft)"] = clean_numeric(sf_df.get("Floor Area"))
    out["Year Built"] = _to_num(sf_df.get("Year Built"))
    out["# of Buildings"] = pd.NA

    out["ENERGY STAR Score"] = _to_num(sf_df.get("ENERGY STAR Score"))

    # Electricity: kWh -> kBtu
    elec_kwh = _to_num(sf_df.get("Electricity Use - Grid Purchase (kWh)"))
    out["Electricity Use (kBtu)"] = elec_kwh * KWH_TO_KBTU

    out["Natural Gas Use (kBtu)"] = _to_num(sf_df.get("Natural Gas Use (kBtu)"))
    out["District Steam Use (kBtu)"] = _to_num(sf_df.get("District Steam Use (kBtu)"))

    out["District Chilled Water Use (kBtu)"] = pd.NA
    out["All Other Fuel Use (kBtu)"] = pd.NA

    # EUI: ft2 == sq ft
    out["Site EUI (kBtu/sq ft)"] = _to_num(sf_df.get("Site EUI (kBtu/ft2)"))
    out["Source EUI (kBtu/sq ft)"] = _to_num(sf_df.get("Source EUI (kBtu/ft2)"))
    out["Weather Normalized Site EUI (kBtu/sq ft)"] = _to_num(
        sf_df.get("Weather Normalized Site EUI (kBtu/ft2)")
    )
    out["Weather Normalized Source EUI (kBtu/sq ft)"] = _to_num(
        sf_df.get("Weather Normalized Source EUI (kBtu/ft2)")
    )

    out["Total GHG Emissions (Metric Tons CO2e)"] = _to_num(
        sf_df.get("Total GHG Emissions (Metric Tons CO2e)")
    )
    out["GHG Intensity (kg CO2e/sq ft)"] = _to_num(
        sf_df.get("Total GHG Emissions Intensity (kGCO2e/ft2)")
    )

    # SF lat/lon are often lowercase
    out["Latitude"] = _to_num(sf_df.get("latitude"))
    out["Longitude"] = _to_num(sf_df.get("longitude"))
    out["Location"] = _make_location(out, "Latitude", "Longitude")

    out["Reporting Status"] = sf_df.get("Benchmark Status")
    out["Chicago Energy Rating"] = pd.NA
    out["Exempt From Chicago Energy Rating"] = sf_df.get("Reason for Exemption")

    out["Water Use (kGal)"] = pd.NA
    out["Row_ID"] = pd.NA

    for c in CHICAGO_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    return out[CHICAGO_COLS].copy()


def seattle_to_chicago(seattle_df: pd.DataFrame) -> pd.DataFrame:
    """Map Seattle benchmarking data (current column-name variant) into Chicago's schema.

    Seattle raw columns look like: OSEBuildingID, DataYear, BuildingName, ZipCode, ...
    """
    out = pd.DataFrame(index=seattle_df.index)

    out["Data Year"] = _to_num(seattle_df.get("DataYear"))
    out["ID"] = seattle_df.get("OSEBuildingID")

    out["Property Name"] = seattle_df.get("BuildingName")
    out["Address"] = seattle_df.get("Address")
    out["ZIP Code"] = seattle_df.get("ZipCode")

    out["Community Area"] = pd.NA
    out["Primary Property Type"] = seattle_df.get("EPAPropertyType")

    # Prefer Buildings GFA; fallback to Total; fallback to self-report
    if "PropertyGFABuildings" in seattle_df.columns:
        out["Gross Floor Area - Buildings (sq ft)"] = clean_numeric(
            seattle_df.get("PropertyGFABuildings")
        )
    elif "PropertyGFATotal" in seattle_df.columns:
        out["Gross Floor Area - Buildings (sq ft)"] = _to_num(
            seattle_df.get("PropertyGFATotal")
        )
    else:
        out["Gross Floor Area - Buildings (sq ft)"] = _to_num(
            seattle_df.get("SelfReportGFABuildings")
        )

    out["Year Built"] = _to_num(seattle_df.get("YearBuilt"))
    out["# of Buildings"] = _to_num(seattle_df.get("NumberofBuildings"))

    out["ENERGY STAR Score"] = _to_num(seattle_df.get("ENERGYSTARScore"))

    # Electricity: prefer kBtu; else convert kWh -> kBtu
    if "Electricity(kBtu)" in seattle_df.columns:
        out["Electricity Use (kBtu)"] = _to_num(seattle_df.get("Electricity(kBtu)"))
    else:
        out["Electricity Use (kBtu)"] = (
            _to_num(seattle_df.get("Electricity(kWh)")) * KWH_TO_KBTU
        )

    # Natural gas: prefer kBtu; else therms -> kBtu (1 therm = 100 kBtu)
    if "NaturalGas(kBtu)" in seattle_df.columns:
        out["Natural Gas Use (kBtu)"] = _to_num(seattle_df.get("NaturalGas(kBtu)"))
    else:
        out["Natural Gas Use (kBtu)"] = (
            _to_num(seattle_df.get("NaturalGas(therms)")) * 100.0
        )

    out["District Steam Use (kBtu)"] = _to_num(seattle_df.get("SteamUse(kBtu)"))

    out["District Chilled Water Use (kBtu)"] = pd.NA
    out["All Other Fuel Use (kBtu)"] = pd.NA

    # EUI units already match Chicago (kBtu/sf == kBtu/sq ft)
    out["Site EUI (kBtu/sq ft)"] = _to_num(seattle_df.get("SiteEUI(kBTu/sf)"))
    if out["Site EUI (kBtu/sq ft)"].isna().all():
        out["Site EUI (kBtu/sq ft)"] = _to_num(seattle_df.get("SiteEUI(kBtu/sf)"))

    out["Source EUI (kBtu/sq ft)"] = _to_num(seattle_df.get("SourceEUI(kBtu/sf)"))
    out["Weather Normalized Site EUI (kBtu/sq ft)"] = _to_num(
        seattle_df.get("SiteEUIWN(kBtu/sf)")
    )
    out["Weather Normalized Source EUI (kBtu/sq ft)"] = _to_num(
        seattle_df.get("SourceEUIWN(kBtu/sf)")
    )

    out["Total GHG Emissions (Metric Tons CO2e)"] = _to_num(
        seattle_df.get("TotalGHGEmissions")
    )
    out["GHG Intensity (kg CO2e/sq ft)"] = _to_num(
        seattle_df.get("GHGEmissionsIntensity")
    )

    out["Latitude"] = _to_num(seattle_df.get("Latitude"))
    out["Longitude"] = _to_num(seattle_df.get("Longitude"))
    out["Location"] = _make_location(out, "Latitude", "Longitude")

    out["Reporting Status"] = seattle_df.get("ComplianceStatus")
    out["Chicago Energy Rating"] = pd.NA
    out["Exempt From Chicago Energy Rating"] = seattle_df.get("ComplianceIssue")

    out["Water Use (kGal)"] = pd.NA
    out["Row_ID"] = pd.NA

    for c in CHICAGO_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    return out[CHICAGO_COLS].copy()


MAJOR_US_CITIES_DIR: Final[Path] = Path("data") / "major_us_cities_data"


def _infer_year_from_name(filename: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", filename)
    if match is None:
        return None
    return int(match.group(0))


def _unnamed_share(columns: pd.Index) -> float:
    cols = columns.astype(str)
    if len(cols) == 0:
        return 1.0
    n_unnamed = cols.str.match(r"^Unnamed").sum()
    return n_unnamed / len(cols)


def _read_boston_excel(path: Path) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    for header in (0, 1, 2, 3, 4, 5):
        dff = pd.read_excel(path, engine="openpyxl", header=header)
        dff.columns = [str(c).strip() for c in dff.columns]
        dff = dff.dropna(how="all")

        candidates.append(dff)

    # Choose the version with the lowest share of Unnamed columns
    best = min(candidates, key=lambda d: _unnamed_share(pd.Index(d.columns)))
    return best


def _read_boston_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            dff = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            dff = pd.read_csv(path, low_memory=False, encoding="latin-1")

        dff.columns = [str(c).strip() for c in dff.columns]
        dff = dff.loc[:, ~pd.Index(dff.columns).astype(str).str.match(r"^Unnamed")]
        dff = dff.dropna(axis=1, how="all").dropna(how="all")
        return dff

    if suffix in {".xlsx", ".xls"}:
        boston_df = _read_boston_excel(path)
        boston_df = boston_df.loc[
            :,
            ~pd.Index(boston_df.columns).astype(str).str.match(r"^Unnamed"),
        ]
        boston_df = boston_df.dropna(axis=1, how="all").dropna(how="all")
        return boston_df

    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_boston_energy_data(
    folder: str | Path | None = None,
    *,
    recursive: bool = False,
    city_name: str = "Boston",
    year_col: str = "Data Year",
) -> pd.DataFrame:
    """Load Boston energy benchmarking data from a folder.

    Loads multiple CSV/XLSX files and returns a single concatenated DataFrame.

    Adds:
      - City
      - source_file
      - Data Year (inferred from filename if missing)
    """
    if folder is None:
        folder_path = MAJOR_US_CITIES_DIR / "Boston_data"
    else:
        folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Boston folder not found: {folder_path.resolve()}")

    glob_pattern = "**/*" if recursive else "*"
    files = sorted(
        list(folder_path.glob(f"{glob_pattern}.csv"))
        + list(folder_path.glob(f"{glob_pattern}.xlsx"))
        + list(folder_path.glob(f"{glob_pattern}.xls"))
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV/XLSX files found in Boston folder: {folder_path.resolve()}"
        )

    def process_boston_file(path: Path) -> pd.DataFrame:
        """Process a single Boston file with metadata."""
        boston_df = _read_boston_file(path)
        boston_df["City"] = city_name
        boston_df["source_file"] = path.name

        if year_col not in boston_df.columns:
            inferred = _infer_year_from_name(path.name)
            if inferred is not None:
                boston_df[year_col] = inferred

        return boston_df

    # Load Boston files in parallel for faster performance
    with ThreadPoolExecutor() as executor:
        frames = list(executor.map(process_boston_file, files))

    out = pd.concat(frames, ignore_index=True, sort=False)

    if year_col in out.columns:
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")

    return out


def _normalize_colname(name: str) -> str:
    """Normalize a column name for matching (lowercase, remove symbols/spaces)."""
    s = name.strip().lower()
    s = re.sub(r"[^\w]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def coalesce_columns(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    *,
    drop_sources: bool = True,
) -> pd.DataFrame:
    """Create canonical columns by taking first non-null across source columns."""
    out = df.copy()

    for target, sources in groups.items():
        existing = [c for c in sources if c in out.columns]
        if not existing:
            continue

        series = out[existing[0]]
        for c in existing[1:]:
            series = series.combine_first(out[c])

        out[target] = series

        if drop_sources:
            drop_cols = [c for c in existing if c != target]
            out = out.drop(columns=drop_cols, errors="ignore")

    return out


def harmonize_boston_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Boston duplicate columns into a single canonical set."""
    groups = {
        # Identifiers
        "Property Name": ["Property Name"],
        "Address": ["Address", "Building Address", "Parcel Address"],
        "ZIP": [
            "ZIP",
            "Zip",
            "Building Address Zip Code",
            "Parcel Address Zip Code",
            "Building Address Zip  Code",
        ],
        # Types
        "Primary Property Type": [
            "Property Type",
            "Reported Property Type",
            "Largest Property Type",
        ],
        # Floor area
        "Gross Floor Area (sq ft)": [
            "Gross Area (sq ft)",
            "Reported Gross Floor Area (Sq Ft)",
        ],
        # EUI
        "Site EUI (kBtu/sq ft)": [
            "Site EUI (kBTU/sf)",
            "Site EUI (kBtu/ft²)",
            "Site EUI (Energy Use Intensity kBTu/ft²)",
            "Site EUI (Energy Use Intensity kBtu/ft²)",
        ],
        # ENERGY STAR
        "ENERGY STAR Score": ["Energy Star Score", "ENERGY STAR Score"],
        "ENERGY STAR Certified": ["Energy Star Certified"],
        # Energy totals
        "Total Site Energy (kBtu)": [
            "Total Site Energy (kBTU)",
            "Total Site Energy Usage (kBtu)",
        ],
        # Water
        "Water Intensity (gal/sq ft)": [
            "Water Intensity (gal/sf)",
            "Water Usage Intensity (Gallons/ft²)",
        ],
        # GHG
        "GHG Emissions (MTCO2e)": ["GHG Emissions (MTCO2e)"],
        "GHG Intensity (kgCO2e/sq ft)": ["GHG Intensity (kgCO2/sf)"],
    }

    out = df.copy()

    # Fix the one common typo in your list
    if (
        "Cooresponding Campus ID" in out.columns
        and "Corresponding Campus ID" not in out.columns
    ):
        out = out.rename(columns={"Cooresponding Campus ID": "Corresponding Campus ID"})

    out = coalesce_columns(out, groups, drop_sources=True)

    # Optional: strip whitespace in key string fields
    for col in ["Property Name", "Address", "ZIP", "Primary Property Type"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().replace({"nan": pd.NA})

    return out


def boston_to_chicago(boston_df: pd.DataFrame) -> pd.DataFrame:
    """Map Boston benchmarking data into Chicago's schema.

    Boston input is your harmonized Boston dataset (after harmonize_boston_columns),
    with columns like: Property Name, Address, ZIP, BERDO ID, Primary Property Type,
    Gross Floor Area (sq ft), Site EUI (kBtu/sq ft), ENERGY STAR Score,
    Electricity Usage (kBtu), Natural Gas Usage (kBtu), District Steam Usage (kBtu),
    District Chilled Water Usage (kBtu), Fuel Oil * Usage (kBtu), etc.
    """
    out = pd.DataFrame(index=boston_df.index)

    # Required Chicago identifiers
    out["Data Year"] = pd.to_numeric(
        boston_df.get("Data Year"), errors="coerce"
    ).astype("Int64")
    out["ID"] = boston_df.get("BERDO ID").combine_first(boston_df.get("Tax Parcel ID"))
    out["Property Name"] = boston_df.get("Property Name")
    out["Address"] = boston_df.get("Address")
    out["ZIP Code"] = boston_df.get("ZIP")
    out["Community Area"] = pd.NA
    out["Primary Property Type"] = boston_df.get("Primary Property Type")

    # Buildings / floor area
    out["Gross Floor Area - Buildings (sq ft)"] = pd.to_numeric(
        boston_df.get("Gross Floor Area (sq ft)"),
        errors="coerce",
    )
    out["Year Built"] = pd.to_numeric(
        boston_df.get("Year Built"), errors="coerce"
    ).astype("Int64")
    out["# of Buildings"] = pd.NA

    # ENERGY STAR + EUI
    out["ENERGY STAR Score"] = pd.to_numeric(
        boston_df.get("ENERGY STAR Score"),
        errors="coerce",
    ).astype("Int64")

    out["Site EUI (kBtu/sq ft)"] = pd.to_numeric(
        boston_df.get("Site EUI (kBtu/sq ft)"),
        errors="coerce",
    )
    out["Source EUI (kBtu/sq ft)"] = pd.NA
    out["Weather Normalized Site EUI (kBtu/sq ft)"] = pd.NA
    out["Weather Normalized Source EUI (kBtu/sq ft)"] = pd.NA

    # Energy by fuel (kBtu)
    out["Electricity Use (kBtu)"] = pd.to_numeric(
        boston_df.get("Electricity Usage (kBtu)"),
        errors="coerce",
    )
    out["Natural Gas Use (kBtu)"] = pd.to_numeric(
        boston_df.get("Natural Gas Usage (kBtu)"),
        errors="coerce",
    )
    out["District Steam Use (kBtu)"] = pd.to_numeric(
        boston_df.get("District Steam Usage (kBtu)"),
        errors="coerce",
    )
    out["District Chilled Water Use (kBtu)"] = pd.to_numeric(
        boston_df.get("District Chilled Water Usage (kBtu)"),
        errors="coerce",
    )

    # Chicago has "All Other Fuel Use" (Boston has several components)
    other_fuels = [
        "District Hot Water Usage (kBtu)",
        "Fuel Oil 1 Usage (kBtu)",
        "Fuel Oil 2 Usage (kBtu)",
        "Fuel Oil 4 Usage (kBtu)",
        "Fuel Oil 5 and 6 Usage (kBtu)",
        "Propane Usage (kBtu)",
        "Diesel Usage (kBtu)",
        "Kerosene Usage (kBtu)",
        "Renewable System Electricity Usage Onsite (kBtu)",
    ]

    present = [c for c in other_fuels if c in boston_df.columns]
    if present:
        other_numeric = boston_df[present].apply(pd.to_numeric, errors="coerce")
        out["All Other Fuel Use (kBtu)"] = other_numeric.sum(axis=1, min_count=1)
    else:
        out["All Other Fuel Use (kBtu)"] = pd.NA

    # GHG
    out["Total GHG Emissions (Metric Tons CO2e)"] = pd.to_numeric(
        boston_df.get("GHG Emissions (MTCO2e)"),
        errors="coerce",
    )
    out["GHG Intensity (kg CO2e/sq ft)"] = pd.to_numeric(
        boston_df.get("GHG Intensity (kgCO2e/sq ft)"),
        errors="coerce",
    )

    # Location fields (not in Boston)
    out["Latitude"] = pd.NA
    out["Longitude"] = pd.NA
    out["Location"] = pd.NA

    # Status / ratings
    out["Reporting Status"] = boston_df.get(
        "Reporting Compliance Status"
    ).combine_first(boston_df.get("Compliance Status"))
    out["Chicago Energy Rating"] = pd.NA
    out["Exempt From Chicago Energy Rating"] = pd.NA

    # Water (Chicago is kGal; Boston is intensity, not total)
    out["Water Use (kGal)"] = pd.NA

    # Row id (not in Boston)
    out["Row_ID"] = pd.NA

    return out
