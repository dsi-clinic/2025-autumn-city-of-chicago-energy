"""Utilities for loading and cleaning Chicago Energy Benchmarking data from CSV files."""

import json
import logging
import re
from pathlib import Path

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

    load_dfs = [pd.read_csv(file) for file in csv_files]
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


def clean_year_built(
    energy_df: pd.DataFrame,
    id_col: str = "ID",
    year_col: str = "Data Year",
    year_built_col: str = "Year Built",
) -> pd.DataFrame:
    """Clean 'Year Built' for each building (ID), sorted by Data Year ascending.

    Rules:
    - Maintain a 'current' Year Built value as we move forward in time.
    - If a row has Year Built == Data Year (rebuild year), set current = Data Year.
    - If a row has a non-null Year Built != Data Year and current is still null,
      use that as the initial current value.
    - For every row, if current is not null, set Year Built = current.
    """
    cleaned_df = energy_df.copy()
    cleaned_df[year_built_col] = pd.to_numeric(
        cleaned_df[year_built_col], errors="coerce"
    )
    cleaned_df[year_col] = pd.to_numeric(cleaned_df[year_col], errors="coerce")

    def fix_building(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(year_col).copy()
        current: float | None = None
        cleaned_values: list[float | None] = []

        for _, row in group.iterrows():
            year = row[year_col]
            year_built = row[year_built_col]

            if pd.notna(year_built) and pd.notna(year) and year_built == year:
                # rebuild in this year: from now on use this year
                current = year_built
            elif current is None and pd.notna(year_built):
                # first known value before any rebuild year
                current = year_built

            cleaned_values.append(current if current is not None else year_built)

        group[year_built_col] = cleaned_values
        return group

    cleaned_df = cleaned_df.groupby(id_col, group_keys=False).apply(fix_building)

    return cleaned_df


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

    return data


def build_compliance_base_year(
    energy_data: pd.DataFrame,
    year: int,
    year_col: str = "Data Year",
    id_col: str = "ID",
    area_col: str = "Community Area",
    property_type_col: str = "Primary Property Type",
    status_col: str = "compliance_status",
) -> pd.DataFrame:
    """One row per unique (building_id, area_key, ptype_norm) for a given year.

    Uses the explicit compliance_status in energy_data.

    Returns columns:
      [id_col, area_key, area_display, ptype_norm, compliance_status]
    """

    def norm_upper(x: str | None) -> str | None:
        return pd.NA if pd.isna(x) else str(x).strip().upper()

    base = energy_data.loc[
        energy_data[year_col] == year,
        [id_col, area_col, property_type_col, status_col],
    ].copy()

    base["_id"] = pd.to_numeric(base[id_col], errors="coerce")
    base["area_key"] = base[area_col].apply(norm_upper)
    base["area_display"] = base[area_col].astype(str).str.strip().str.title()
    base[status_col] = base[status_col].astype(str).str.strip().str.lower()

    base = base.dropna(subset=["_id", "area_key", "Primary Property Type", status_col])
    base["_id"] = base["_id"].astype(int)

    # One row per (building, area, property type) within year
    base = base.drop_duplicates(
        subset=["_id", "area_key", "Primary Property Type"]
    ).copy()

    return base.rename(columns={"_id": "Building ID"})[
        ["Building ID", "area_key", "area_display", "Primary Property Type", status_col]
    ]


def build_area_table_overall(
    base: pd.DataFrame,
    status_col: str = "compliance_status",
) -> pd.DataFrame:
    """Community-area table (overall)."""
    data = base.copy()
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

    counts = counts.rename(columns={"non-compliant": "non_compliant"})
    counts["denom"] = counts["compliant"] + counts["non_compliant"]
    counts["non_compliance_rate"] = counts["non_compliant"] / counts["denom"].replace(
        0, pd.NA
    )

    return counts[
        [
            "area_key",
            "area_display",
            "compliant",
            "non_compliant",
            "exempt",
            "denom",
            "non_compliance_rate",
        ]
    ]


def build_area_table_by_property(
    base: pd.DataFrame,
    top_n_property_types: int = 10,
    id_col: str = "Building ID",
    ptype_col: str = "Primary Property Type",
    status_col: str = "compliance_status",
) -> tuple[pd.DataFrame, list[str]]:
    """Area × property type table for dropdown-filter choropleth.

    Returns:
      - area_type table with:
          area_key, area_display, ptype_key,
          compliant, non_compliant, denom, non_compliance_rate,
          _lookup_key (= area_key + '|' + ptype_key)
      - list of top property types (ptype_key)
    """
    data = base.copy()
    data[status_col] = data[status_col].astype(str).str.strip().str.lower()
    data = data[data[status_col].isin(["compliant", "non-compliant"])].copy()

    top_ptypes = (
        data.groupby(ptype_col)[id_col]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n_property_types)
        .index.tolist()
    )

    data = data[data[ptype_col].isin(top_ptypes)].copy()

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
            "non-compliant": "non_compliant",
        }
    )

    area_type["denom"] = area_type["compliant"] + area_type["non_compliant"]
    area_type["non_compliance_rate"] = area_type["non_compliant"] / area_type[
        "denom"
    ].replace(0, pd.NA)

    area_type["_lookup_key"] = area_type["area_key"] + "|" + area_type["ptype_key"]

    return area_type[
        [
            "area_key",
            "area_display",
            "ptype_key",
            "compliant",
            "non_compliant",
            "denom",
            "non_compliance_rate",
            "_lookup_key",
        ]
    ], top_ptypes


def compliance_by_category(
    energy_data: pd.DataFrame,
    year: int,
    category_col: str,
    year_col: str = "Data Year",
    id_col: str = "ID",
    status_col: str = "compliance_status",
) -> pd.DataFrame:
    """Compute compliance and non-compliance rates by a categorical variable (e.g., property type, sector) for a given year.

    Excludes 'exempt' from denominator.
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
