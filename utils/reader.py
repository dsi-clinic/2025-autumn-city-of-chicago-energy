"""reader.py

This module provides utilities for reading and consolidating energy benchmarking data.
It includes a function for combining Chicago Energy Benchmarking data which returns the combined DataFrame
as well as saving the combined data as a .csv
"""

import re
from pathlib import Path

import pandas as pd


def combine_energy_data(
    data_folder: str = "../data/chicago_energy_benchmarking",
    output_path: str = "../output/combined_energy_benchmarking.csv",
) -> pd.DataFrame:
    """Combines all yearly Chicago Energy Benchmarking CSV files (2014–2023) into a single DataFrame.
    
    Expects files in the format:
        Chicago_Energy_Benchmarking_-_2014_Data_Reported_in_2015_20251002.csv
        ...
        Chicago_Energy_Benchmarking_-_2023_Data_Reported_in_2024_20251002.csv
    located in the `data/` directory.

    Parameters
    ----------
    data_folder : str
        Path to the directory containing the input CSV files.
    output_path : str
        Path where the combined CSV should be saved.

    Returns:
    -------
    pd.DataFrame
        Combined DataFrame with all years of benchmarking data.
    """
    data_folder_path = Path(data_folder)
    file_list = sorted(
        data_folder_path.glob("Chicago_Energy_Benchmarking_-_*_Data_Reported_in_*_20251002.csv")
    )

    if not file_list:
        print(f"No matching CSV files found in {data_folder}.")
        return None

    all_dfs = []
    for file_path in file_list:
        match = re.search(r"Benchmarking_-_(\d{4})_Data", file_path.name)
        year = int(match.group(1)) if match else None

        print(f"Reading {file_path.name} (Year: {year})...")
        energy = pd.read_csv(file_path)

        # Add year column
        energy["Benchmark_Year"] = year
        all_dfs.append(energy)

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(
        f"Combined {len(file_list)} files with {combined_df.shape[0]:,} total rows."
    )

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save combined data
    combined_df.to_csv(output_path, index=False)
    print(f"💾 Saved combined data to: {output_path}")

    return combined_df


if __name__ == "__main__":
    combine_energy_data()
