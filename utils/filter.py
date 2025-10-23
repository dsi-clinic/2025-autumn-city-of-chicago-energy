"""filter.py

This module filters the combined dataset from yearly Chicago Energy Benchmarking
datasets (2014–2023) and retrives the buildings with data reported for the required
year range / number of years.
"""

import pandas as pd


def filter_buildings_by_year_range(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    id_col: str = "ID",
    year_col: str = "Data Year",
) -> pd.DataFrame:
    """Filter buildings that have submitted data for all years in a specified range.

    Parameters
    ----------
    df : pd.DataFrame
        The energy dataset containing at least building ID and year columns.
    start_year : int
        The first year in the required range (inclusive).
    end_year : int
        The last year in the required range (inclusive).
    id_col : str, default="ID"
        The column name that identifies unique buildings.
    year_col : str, default="Data Year"
        The column name indicating the year of the data entry.

    Returns:
    -------
    pd.DataFrame
        A filtered DataFrame containing only records of buildings that have
        data submitted for all years in the specified range.
    """
    required_years = set(range(start_year, end_year + 1))

    # Group by building ID and collect unique years
    building_years = df.groupby(id_col)[year_col].unique().reset_index(name="Years")

    # Keep only those with full year coverage
    buildings_all_years = building_years[
        building_years["Years"].apply(lambda years: required_years.issubset(set(years)))
    ]

    # Filter original DataFrame
    filtered_df = df[df[id_col].isin(buildings_all_years[id_col])].copy()

    # To make sure the rows are unique without duplicates
    filtered_df = filtered_df.drop_duplicates(subset=["ID", "Data Year"], keep="first")

    return filtered_df


def pivot_energy_metric(
    df: pd.DataFrame,
    metric_col: str,
    start_year: int,
    end_year: int,
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
    start_year : int
        The first year in the building range to consider for dropping nulls.
    end_year : int
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
