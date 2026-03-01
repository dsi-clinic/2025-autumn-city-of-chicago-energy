"""Utilities for loading and cleaning Chicago Energy Benchmarking data from CSV files."""

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from utils.settings import DATA_DIR

logger = logging.getLogger(__name__)

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

    # Load and concatenate all CSVs
    load_dfs = [pd.read_csv(file) for file in csv_files]
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

    Parameters
    ----------
    df : pd.DataFrame
        The energy dataset containing at least building ID and year columns.
    start_year : int, default = 2016
        The first year in the required range (inclusive).
    end_year : int, default = 2023
        The last year in the required range (inclusive).
    id_col : str, default="ID"
        The column name that identifies unique buildings.
    year_col : str, default="Data Year"
        The column name indicating the year of the data entry.
    building_type_col : str, default="Primary Property Type"
        The column name for the building type.
    building_type : list, default=[]
        A list of building types to include. If empty, all types are included.

    Returns:
    -------
    pd.DataFrame
        A filtered DataFrame containing only records of buildings that have
        data submitted for all years in the specified range, restricted to data within that range.

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


def clean_property_type(energy_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure each building (ID) has a consistent Primary Property Type.

    Rules:
    1. If a building has only one valid (non-'nan'/non-empty) type, fill all with that.
    2. If a building has any combination of 'multifamily housing', 'residential', or 'nan',
       set all to 'multifamily housing'.
    3. If a building only has repeated instances of 'multifamily housing', keep that.
    4. If a building has any combination of 'senior care community' or 'senior living community',
       set all to 'senior care community'.
    """
    df_copy = energy_df.copy()

    missing_vals = {"nan", "none", ""}

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

    # Map each building ID to its valid property types
    type_map = df_copy.groupby("ID")["Primary Property Type"].apply(
        lambda x: [
            re.sub(r"\s+", " ", str(v)).strip().lower()
            for v in x
            if pd.notna(v) and str(v).strip().lower() not in missing_vals
        ]
    )

    id_to_type = {}

    for bid, types in type_map.items():
        lower_types = {t.strip().lower() for t in types}

        # Case 4: senior care / senior living -> unify as 'senior care community'
        if lower_types & {"senior care community", "senior living community"}:
            id_to_type[bid] = "senior care community"

        # Case 1: Only one valid type
        elif len(lower_types) == 1:
            id_to_type[bid] = list(lower_types)[0]

        # Case 2: multifamily + residential/nan/duplicates -> multifamily housing
        elif "multifamily housing" in lower_types and (
            "residential" in lower_types or "nan" in lower_types or "" in lower_types
        ):
            id_to_type[bid] = "multifamily housing"

        # Case 3: redundant duplicates like ['multifamily housing', 'multifamily housing']
        elif lower_types & {"multifamily housing"}:
            id_to_type[bid] = "multifamily housing"

        # Case 5: mall types -> unify as 'mall'
        if lower_types & {"enclosed mall", "strip mall", "other - mall"}:
            id_to_type[bid] = "mall"

        # Case 6: residential types -> unify as 'residential'
        if lower_types & {"residential", "other - lodging/residential"}:
            id_to_type[bid] = "residential"

        # Case 7: hospital types -> unify as 'hospital'
        if lower_types & {
            "hospital (general medical & surgical)",
            "other - specialty hospital",
        }:
            id_to_type[bid] = "hospital"

        # Case 8: other - recreation -> recreation
        if lower_types & {"other - recreation"}:
            id_to_type[bid] = "recreation"

        if lower_types & merge_to_other:
            id_to_type[bid] = "other"

    # Apply replacements
    def replace_type(row: pd.Series) -> str:
        val = str(row["Primary Property Type"]).strip().lower()
        if val in missing_vals or pd.isna(row["Primary Property Type"]):
            return id_to_type.get(row["ID"], row["Primary Property Type"])
        if row["ID"] in id_to_type:
            return id_to_type[row["ID"]]
        return row["Primary Property Type"]

    df_copy["Primary Property Type"] = df_copy.apply(replace_type, axis=1)

    return df_copy


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


def assign_effective_year_built(buildings: pd.DataFrame) -> pd.DataFrame:
    """Add an 'Effective Year Built' column per building ID.

    Rules (computed within each ID group):
    - If there is exactly one unique non-null Year Built, assign that year to all rows.
    - If there are multiple unique non-null Year Built values, assign "Multiple Years Built".
    - If all Year Built values are null, assign NaN.

    Parameters
    ----------
    buildings:
        DataFrame containing at least 'ID' and 'Year Built'.

    Returns:
    -------
    pd.DataFrame
        Copy of input with an added 'Effective Year Built' column.
    """
    out = buildings.copy()

    # Count distinct, non-null years per ID
    n_unique = out.groupby("ID")["Year Built"].transform(lambda s: s.dropna().nunique())

    # The single (non-null) year, repeated per row (NaN if none)
    single_year = out.groupby("ID")["Year Built"].transform("first")

    # Build result
    out["Effective Year Built"] = np.where(
        n_unique == 1,
        single_year,
        np.where(n_unique > 1, "Multiple Years Built", np.nan),
    )

    return out


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

    for file in csv_files:
        df_city = pd.read_csv(file, low_memory=False)
        key = file.stem
        city_data[key] = df_city
        logger.info("Loaded %s → %s", file.name, df_city.shape)

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

    frames: list[pd.DataFrame] = []
    for path in files:
        boston_df = _read_boston_file(path)
        boston_df["City"] = city_name
        boston_df["source_file"] = path.name

        if year_col not in boston_df.columns:
            inferred = _infer_year_from_name(path.name)
            if inferred is not None:
                boston_df[year_col] = inferred

        frames.append(boston_df)

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
