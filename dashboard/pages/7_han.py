"""Han Test Page"""

import streamlit as st
import altair as alt

from utils.data_utils import (
    assign_effective_year_built,
    categorize_time_built,
    clean_property_type,
    concurrent_buildings,
    load_data,
    prepare_persistence,
)
from utils.plot_utils import plot_energy_persistence_rows

st.title("Energy Persistence Visualization")

energy_df = load_data()
energy_df = assign_effective_year_built(energy_df)
energy_df = concurrent_buildings(energy_df, 2016, 2023)
energy_df = clean_property_type(energy_df)
energy_df = categorize_time_built(energy_df)

variables = [
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
]
string_columns = ["Time Built", "Primary Property Type", "Community Area"]

category_col = st.selectbox(
    "Select category for Building Classification",
    options=string_columns,
    index=string_columns.index("Decade Built") if "Decade Built" in string_columns else 0
)

site_eui_col = st.selectbox(
    "Select column for Energy Metric",
    options=variables,
    index=variables.index("Site EUI (kBtu/sq ft)") if "Site EUI (kBtu/sq ft)" in variables else 0
)

df_lagged = prepare_persistence(
    energy_df,
    decade_built_col=category_col,
    site_eui_col=site_eui_col,
)

options = sorted(df_lagged[category_col].unique().tolist())
selected_category = st.selectbox(category_col, options)
rows = plot_energy_persistence_rows(
    df_lagged=df_lagged,
    property_col=category_col,  
    id_col="ID",
    year_col="Data Year",
    delta_col="Delta",          
    delta_next_col="Delta_next",  
    start_year=2017,
    end_year=2023,
    selected_category= selected_category,
)

for row_chart in rows:
    st.altair_chart(row_chart, use_container_width=True)
