from pathlib import Path
import pandas as pd
import numpy as np
def testing_loader():
    return print("found data loader file!")

def loading_data():
    target_folder="src"
    current = Path.cwd()
    while current.name != target_folder:
        if current.parent == current:
            raise FileNotFoundError(f"Could not find folder named '{target_folder}' above {Path.cwd()}")
        current = current.parent

    path = current / "data" / "chicago_energy_benchmarking"
    print(path)

    load_dfs = []
    for file in path.rglob("*.csv"):
        print("Reading:", file)
        load_dfs.append(pd.read_csv(file))
    full_df = pd.concat(load_dfs, axis=0, join='outer', ignore_index=True)
    full_df = full_df.sort_values(by="Data Year", ascending=True)

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

    full_df[str_cols] = full_df[str_cols].astype(str)

    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = (
                full_df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"": np.nan})     
                .astype("float64", errors="ignore")
            )
    return full_df
