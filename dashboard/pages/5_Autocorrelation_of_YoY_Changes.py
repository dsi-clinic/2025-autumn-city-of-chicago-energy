"""Building Year to Year Difference Visualization"""

import streamlit as st

from utils.dashboard_utils import (
    apply_category_filter,
    apply_page_config,
    build_standard_filters,
    filter_energy_by_selections,
    load_clean_energy_data_for_dashboards,
)
from utils.data_utils import (
    add_top_level_property_type,
    prepare_persistence,
)
from utils.plot_utils import plot_energy_persistence_rows

"""Streamlit page for analyzing autocorrelation of year-over-year energy changes."""
apply_page_config()
st.title("Autocorrelation of Year-over-Year Changes in Energy Use")
st.markdown("""
When analyzing trends over time, examining the relationship between past and future at the building level (autocorrelation) can provide insight into what is driving aggregate changes over time. Specifically, this analysis examines the lag-1 first-difference autocorrelation: if a building saw an increase in energy use last year, is it more or less likely to show an increase this year—or vice versa?

Each chart below shows the relationship between the year-over-year change in energy use (Δ) of a building in a given year (e.g. 2016) and in the following year (e.g. 2017). Use the dropdowns to group or filter by year built, property type, and community area.
""")

# Load and clean data using standard dashboard utilities
energy_df = load_clean_energy_data_for_dashboards(
    restrict_to_concurrent=True, concurrent_start=2016, concurrent_end=2023
)
energy_df = add_top_level_property_type(energy_df)

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

# Standard filters using dashboard_utils (with Top Level Property Type)
category_col, sel_time_built, sel_ppt, sel_tlpt, sel_ca = build_standard_filters(
    energy_df,
    include_top_level=True,  # Include Top Level Property Type
    page_prefix="persistence",
)

site_eui_col = st.selectbox(
    "Select column for Energy Metric",
    options=variables,
    index=variables.index("Site EUI (kBtu/sq ft)"),
    key="persistence_metric",
)

# Apply standard filters
energy_df_filtered = filter_energy_by_selections(
    energy_df,
    sel_time_built=sel_time_built,
    sel_ppt=sel_ppt,
    sel_ca=sel_ca,
    sel_tlpt=sel_tlpt,
)

if energy_df_filtered.empty:
    st.warning(
        "No buildings match the selected filters. Please broaden your selections."
    )
    st.stop()

min_years = 3
if energy_df_filtered["Data Year"].nunique() < min_years:
    st.warning(
        "Not enough years of data for the selected filters to compute year‑to‑year changes."
    )
    st.stop()

# Build lagged dataset
df_lagged = prepare_persistence(
    energy_df_filtered,
    decade_built_col=category_col,
    site_eui_col=site_eui_col,
)

# Category filter (All or single category)
df_lagged_filtered, selected_class_name = apply_category_filter(df_lagged, category_col)

# Add subtitle showing current selections
st.markdown(
    f"**Showing: {selected_class_name}**"
    if selected_class_name != "All"
    else "**Showing: All categories**"
)

# Generate and display charts
rows = plot_energy_persistence_rows(
    df_lagged=df_lagged_filtered,
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
