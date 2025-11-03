"""Initial Explore Page for dashboard"""

import time

import matplotlib.pyplot as plt
import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    build_all_aggregates,
    build_all_year_charts,
    get_energy_data,
    get_geojson,
)
from utils.plot_utils import (
    plot_building_count_map,
    plot_choropleth,
    plot_trend_by_year,
)

# -------------------- Page Setup --------------------
apply_page_config()
start = time.time()
st.title("Exploratory Dashboard")


# -------------------- Load Data --------------------
energy_data = get_energy_data()
geojson_data = get_geojson()

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

# # -------------------- Precompute All-Year Charts --------------------
agg_energy_data = build_all_aggregates(energy_data, variables)
all_year_charts = build_all_year_charts(agg_energy_data, geojson_data)

# -------------------- Trend Plots --------------------
col1, col2 = st.columns(2)

with col1:
    selected1 = st.selectbox(
        "Choose first metric:", variables[1:], key="trend_top_first"
    )
    plot_trend_by_year(energy_data, [selected1], "mean")
    st.pyplot(plt)

with col2:
    # Set default index to the next available option, but keep all options
    default_index = (variables[1:].index(selected1) + 1) % len(variables[1:])
    selected2 = st.selectbox(
        "Choose second metric:",
        variables[1:],
        index=default_index,
        key="trend_top_second",
    )
    plot_trend_by_year(energy_data, [selected2], "mean")
    st.pyplot(plt)


st.divider()
# -------------------- Map Selection --------------------
st.subheader("Maps")

available_years = sorted(energy_data["Data Year"].dropna().unique())
year_options = ["Average (All Years)"] + sorted(
    [int(year) for year in available_years], reverse=True
)

with st.container():
    VarCol, BuildCol = st.columns(2)

    with VarCol:
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                map_select = st.selectbox(
                    "Choose metric for map:", variables, key="map_metric"
                )
            with col2:
                year_select = st.selectbox("Choose year:", year_options, key="year1")
                year_arg = (
                    None if year_select == "Average (All Years)" else int(year_select)
                )

        # Use cached chart if year is None, otherwise recompute chart using cached aggregation
        if year_arg is None:
            map_chart = all_year_charts[map_select]
        else:
            agg = agg_energy_data[map_select]  # reuse cached aggregation
            map_chart = plot_choropleth(geojson_data, agg, map_select, year_arg)
            st.write("making new graph")

        st.altair_chart(map_chart, width="stretch")

    with BuildCol:
        energy_data["Community Area"] = (
            energy_data["Community Area"].astype(str).str.strip().str.title()
        )
        map_build = plot_building_count_map(geojson_data, energy_data)

        st.altair_chart(map_build, width="stretch")

st.write(f"Render time: {time.time() - start:.2f} seconds")
