"""Building Year to Year Difference Visualization"""

import streamlit as st

from utils.data_utils import (
    assign_effective_year_built,
    categorize_time_built,
    clean_property_type,
    concurrent_buildings,
    load_data,
    prepare_persistence,
)
from utils.plot_utils import plot_energy_persistence_rows

st.title("Building Year to Year Difference Visualization")

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

# This is the *facet* dimension (which chart grouping to show)
category_options = ["Time Built", "Primary Property Type", "Community Area"]
category_col = st.selectbox(
    "Select category for Building Classification",
    options=category_options,
    index=category_options.index("Time Built"),
)

site_eui_col = st.selectbox(
    "Select column for Energy Metric",
    options=variables,
    index=variables.index("Site EUI (kBtu/sq ft)"),
)

# Build lagged dataset
df_lagged = prepare_persistence(
    energy_df,
    decade_built_col=category_col,  # just used as grouping key inside your function
    site_eui_col=site_eui_col,
)

# --- Global filters (always visible) ---
col1, col2, col3 = st.columns(3)

with col1:
    time_built_opts = sorted(energy_df["Time Built"].dropna().unique().tolist())
    sel_time_built = st.multiselect(
        "Time Built", time_built_opts, default=time_built_opts
    )

with col2:
    ppt_opts = sorted(energy_df["Primary Property Type"].dropna().unique().tolist())
    sel_ppt = st.multiselect("Primary Property Type", ppt_opts, default=ppt_opts)

with col3:
    ca_opts = sorted(energy_df["Community Area"].dropna().unique().tolist())
    sel_ca = st.multiselect("Community Area", ca_opts, default=ca_opts)

energy_df_filtered = energy_df[
    energy_df["Time Built"].isin(sel_time_built)
    & energy_df["Primary Property Type"].isin(sel_ppt)
    & energy_df["Community Area"].isin(sel_ca)
]

if energy_df_filtered.empty:
    st.warning(
        "No buildings match the selected filters. Please broaden your selections."
    )
    st.stop()

three = 3
if energy_df_filtered["Data Year"].nunique() < three:
    st.warning(
        "Not enough years of data for the selected filters to compute year‑to‑year changes."
    )
    st.stop()

df_lagged = prepare_persistence(
    energy_df_filtered,
    decade_built_col=category_col,
    site_eui_col=site_eui_col,
)

class_opts = sorted(df_lagged[category_col].dropna().unique().tolist())
selected_class = st.selectbox(f"{category_col} filter", ["All"] + class_opts)

if selected_class != "All":
    df_lagged_plot = df_lagged[df_lagged[category_col] == selected_class]
else:
    df_lagged_plot = df_lagged

rows = plot_energy_persistence_rows(
    df_lagged=df_lagged_plot,
    property_col=category_col,
    id_col="ID",
    year_col="Data Year",
    delta_col="Delta",
    delta_next_col="Delta_next",
    start_year=2017,
    end_year=2023,
)

for row_chart in rows:
    st.altair_chart(row_chart, use_container_width=True)