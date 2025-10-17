from pathlib import Path
import pandas as pd
def loading_data():
    target_folder="src"
    current = Path.cwd()
    while current.name != target_folder:
        if current.parent == current:
            raise FileNotFoundError(f"Could not find folder named '{target_folder}' above {Path.cwd()}")
        current = current.parent

    path = current / "data" / "chicago_energy_benchmarking"
    print(path)

    dfs = []
    for file in path.rglob("*.csv"):
        print("Reading:", file)
        dfs.append(pd.read_csv(file))
    full_df = pd.concat(dfs, axis=0, join='outer', ignore_index=True).sort_values(by="Data Year", ascending=True)
    
    full_df['Property Name'] = full_df['Property Name'].astype(str)
    full_df['ZIP Code'] = full_df['ZIP Code'].astype(str)
    full_df['Community Area'] = full_df['Community Area'].astype(str)
    full_df['Primary Property Type'] = full_df['Primary Property Type'].astype(str)
    full_df['Gross Floor Area - Buildings (sq ft)'] = full_df['Gross Floor Area - Buildings (sq ft)'].str.replace(',', '').astype(float)
    full_df['Electricity Use (kBtu)'] = full_df['Electricity Use (kBtu)'].str.replace(',', '').astype(float)
    full_df['Natural Gas Use (kBtu)'] = full_df['Natural Gas Use (kBtu)'].str.replace(',', '').astype(float)
    full_df['District Steam Use (kBtu)'] = full_df['District Steam Use (kBtu)'].str.replace(',', '').astype(float)
    full_df['District Chilled Water Use (kBtu)'] = full_df['District Chilled Water Use (kBtu)'].str.replace(',', '').astype(float)
    full_df['All Other Fuel Use (kBtu)'] = full_df['All Other Fuel Use (kBtu)'].str.replace(',', '').astype(float)
    full_df['Site EUI (kBtu/sq ft)'] = full_df['Site EUI (kBtu/sq ft)'].astype(str).str.replace(',', '').astype(float)
    full_df['Source EUI (kBtu/sq ft)'] = full_df['Source EUI (kBtu/sq ft)'].astype(str).str.replace(',', '').astype(float)
    full_df['Weather Normalized Site EUI (kBtu/sq ft)'] = full_df['Weather Normalized Site EUI (kBtu/sq ft)'].astype(str).str.replace(',', '').astype(float)
    full_df['Weather Normalized Source EUI (kBtu/sq ft)'] = full_df['Weather Normalized Source EUI (kBtu/sq ft)'].astype(str).str.replace(',', '').astype(float)
    full_df['Total GHG Emissions (Metric Tons CO2e)'] = full_df['Total GHG Emissions (Metric Tons CO2e)'].astype(str).str.replace(',', '').astype(float)
    full_df['Water Use (kGal)'] = full_df['Water Use (kGal)'].astype(str).str.replace(',', '').astype(float)

    full_df['Location'] = full_df['Location'].astype(str)
    full_df['Reporting Status'] = full_df['Reporting Status'].astype(str)
    full_df['Exempt From Chicago Energy Rating'] = full_df['Exempt From Chicago Energy Rating'].astype(str)
    full_df['Row_ID'] = full_df['Row_ID'].astype(str)

    return full_df