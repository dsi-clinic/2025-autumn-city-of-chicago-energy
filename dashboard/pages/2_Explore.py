"""Initial Explore Page for dashboard"""

import time

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils.dashboard_utils import apply_page_config
from utils.data_utils import concurrent_buildings, load_neighborhood_geojson
from utils.plot_utils import (
    aggregate_metric,
    plot_choropleth,
    plot_trend_by_year,
)

# -------------------- Page Setup --------------------
apply_page_config()
start = time.time()
st.title("Exploratory Dashboard")


# -------------------- Load Data --------------------
@st.cache_data
def get_energy_data() -> pd.DataFrame:
    """Caching main data"""
    return concurrent_buildings()


@st.cache_data
def get_geojson() -> dict:
    """Caching geojson data"""
    return load_neighborhood_geojson(
        "/project/src/data/chicago_geo/neighborhood_chi.geojson"
    )


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


# -------------------- Precompute All-Year Charts --------------------
@st.cache_data
def build_all_year_charts(
    df: pd.DataFrame, geojson: dict, metrics: list[str]
) -> dict[str, alt.Chart]:
    """Caching altair graphs of each metric in variable list"""
    charts = {}
    for metric in metrics:
        agg = aggregate_metric(df, metric)
        chart = plot_choropleth(geojson, agg, metric, year=None)
        charts[metric] = chart
    return charts


all_year_charts = build_all_year_charts(energy_data, geojson_data, variables)

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
    col1, col2 = st.columns(2)

    with col1:
        map_select = st.selectbox("Choose metric for map:", variables, key="map_metric")
    with col2:
        year_select = st.selectbox("Choose year:", year_options, key="year1")
        year_arg = None if year_select == "Average (All Years)" else int(year_select)

agg_energy_data = {
    metric: aggregate_metric(energy_data, metric) for metric in variables
}

# Plot and display
map = plot_choropleth(geojson_data, agg_energy_data[map_select], map_select, year_arg)
st.altair_chart(map, width="stretch")
st.write(f"Render time: {time.time() - start:.2f} seconds")
