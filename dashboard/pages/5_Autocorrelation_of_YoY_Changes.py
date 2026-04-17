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
st.title("Energy Use Patterns Over Time")

st.markdown("""
### Do Buildings Follow Energy Trends?

When a building's energy use increases one year, what happens the next year?
Does it keep increasing, or does it bounce back?

**This page examines:** Whether buildings show consistent energy use patterns over time,
or whether changes from year to year are random.

**The data:** Covers buildings that reported every year from 2016-2023, allowing us to
track year-over-year changes.

Use the filters below to explore patterns by building type, age, and neighborhood.
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

with st.container(border=True):
    st.markdown(
        '<p style="color:#1e3a5f;font-weight:600;font-size:1rem;margin:0 0 0.5rem 0;">Customize Your View</p>',
        unsafe_allow_html=True,
    )

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
