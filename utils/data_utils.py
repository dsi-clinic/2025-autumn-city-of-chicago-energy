"""Utilities for loading and cleaning Chicago Energy Benchmarking data from CSV files."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def clean_numeric(series: pd.Series) -> pd.Series:
    """Cleaning columns to be numeric data type"""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan})
        .astype("float64", errors="ignore")
    )


def find_src_root(start_path: Path) -> Path:
    """Finds src path in environment"""
    for parent in [start_path] + list(start_path.parents):
        if parent.name == "src":
            return parent
    raise FileNotFoundError("No 'src' directory found in path hierarchy.")


def load_data() -> pd.DataFrame:
    """Load and clean Chicago Energy Benchmarking data from CSV files located in DATA_DIR."""
    src_root = find_src_root(Path.cwd().resolve())
    path = src_root / "data" / "chicago_energy_benchmarking"
    logging.debug(f"Path: {path}")

    load_dfs = []
    for file in path.rglob("*.csv"):
        logging.debug(f"Reading: {file}")
        load_dfs.append(pd.read_csv(file))
    full_df = pd.concat(load_dfs, ignore_index=True)
    full_df = full_df.sort_values(by="Data Year")

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

    full_df[str_cols] = full_df[str_cols].astype(str).apply(lambda col: col.str.lower())

    full_df = full_df.assign(
        **{
            col: clean_numeric(full_df[col])
            for col in numeric_cols
            if col in full_df.columns
        }
    )

    return full_df


# Load in energy data to be used as default value for functions
energy_data = load_data()


def concurrent_buildings(
    df: pd.DataFrame = energy_data,
    start_year: int = 2016,
    end_year: int = 2023,
    id_col: str = "ID",
    year_col: str = "Data Year",
    building_type_col: str = "Primary Property Type",
    building_type: list = None,
) -> pd.DataFrame:
    """Filter buildings that have submitted data for all years in a specified range, keeping only records within that range.

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
    """
    required_years = set(range(start_year, end_year + 1))

    # Restrict to years within the desired range first
    df_in_range = df[(df[year_col] >= start_year) & (df[year_col] <= end_year)]

    # Optionally filter by building type if list is provided
    if building_type:
        df_in_range = df_in_range[df_in_range[building_type_col].isin(building_type)]

    # Group by building ID and collect unique years
    building_years = (
        df_in_range.groupby(id_col)[year_col].unique().reset_index(name="Years")
    )

    # Keep only those with full year coverage
    buildings_all_years = building_years[
        building_years["Years"].apply(lambda years: required_years.issubset(set(years)))
    ]

    # Filter the dataset to only those buildings and within year range
    filtered_df = df_in_range[
        df_in_range[id_col].isin(buildings_all_years[id_col])
    ].copy()

    # Ensure no duplicates (e.g., multiple entries for same building-year)
    filtered_df = filtered_df.drop_duplicates(subset=[id_col, year_col], keep="first")

    return filtered_df


def pivot_energy_metric(
    metric_col: str,
    df: pd.DataFrame = energy_data,
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
    # Create pivot table
    pivot_df = df.pivot_table(index=id_col, columns=year_col, values=metric_col)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
