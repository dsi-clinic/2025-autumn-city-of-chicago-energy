"""data.py

This module merges all yearly Chicago Energy Benchmarking datasets (2014–2023)
and converts numeric columns to usable float64 types.
It automatically handles differences in columns across years and missing values.
"""

import re
from pathlib import Path

import pandas as pd


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns in Chicago Energy Benchmarking datasets to float64.

    Handles missing columns and string artifacts (commas, spaces, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    -------
    pd.DataFrame
        DataFrame with cleaned numeric columns.
    """
    cols_to_convert = [
        "Gross Floor Area - Buildings (sq ft)",
        "Water Use (kGal)",  # added for post-2019 datasets
        "Electricity Use (kBtu)",
        "Natural Gas Use (kBtu)",
        "District Steam Use (kBtu)",
        "District Chilled Water Use (kBtu)",
        "All Other Fuel Use (kBtu)",
        "Total GHG Emissions (Metric Tons CO2e)",
        "GHG Intensity (kg CO2e/sq ft)",
        "Site EUI (kBtu/sq ft)",
        "Source EUI (kBtu/sq ft)",
        "Weather Normalized Site EUI (kBtu/sq ft)",
        "Weather Normalized Source EUI (kBtu/sq ft)",
    ]

    cols_present = [c for c in cols_to_convert if c in df.columns]

    for col in cols_present:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def merge_all_years(
    data_folder: str = "../data/chicago_energy_benchmarking",
    output_path: str = "..output/merged_data.csv",
) -> pd.DataFrame:
    """Merge all yearly Chicago Energy Benchmarking CSVs (2014–2023),

    standardize column names, and convert numeric columns to usable types.

    Parameters
    ----------
    data_folder : str
        Directory containing the CSV files.
    output_path : str
        File path where the merged CSV will be saved.

    Returns:
    -------
    pd.DataFrame
        Cleaned and merged DataFrame.
    """
    data_folder_path = Path(data_folder)
    file_list = sorted(
        data_folder_path.glob(
            "Chicago_Energy_Benchmarking_-_*_Data_Reported_in_*_20251002.csv"
        )
    )

    if not file_list:
        print(f"No matching CSV files found in {data_folder}.")
        return None

    all_dfs = []
    for file_path in file_list:
        match = re.search(r"Benchmarking_-_(\d{4})_Data", file_path.name)
        year = int(match.group(1)) if match else None

        print(f"Reading {file_path.name} (Year: {year})...")
        year_df = pd.read_csv(file_path)

        # Add year
        year_df["Data_Year"] = year

        # Convert numeric columns safely
        year_df = convert_data_types(year_df)

        all_dfs.append(year_df)

    # Align all columns (outer join ensures no data loss)
    merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)

    print(
        f"Merged {len(file_list)} files with {merged_df.shape[0]:,} total rows and {merged_df.shape[1]} columns."
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"Saved cleaned, merged dataset to: {output_path}")

    return merged_df


if __name__ == "__main__":
    merge_all_years()
