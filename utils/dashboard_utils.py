"""Modularizing configurations for each page"""

import json

import altair as alt
import geopandas as gpd
import pandas as pd
import streamlit as st
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.data_utils import concurrent_buildings, load_data, load_neighborhood_geojson
from utils.plot_utils import aggregate_metric, plot_building_count_map, plot_choropleth

# Page layout #-------------------------------------------------------------------


def apply_page_config() -> None:
    """Standardizing page apperance"""
    st.set_page_config(
        page_title="Chicago Energy Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# Caching Data #-------------------------------------------------------------------


@st.cache_data
def cache_full_data() -> pd.DataFrame:
    """Caching main data"""
    full_data = load_data()
    full_data["Community Area"] = (
        full_data["Community Area"].astype(str).str.strip().str.title()
    )
    return full_data


@st.cache_data
def cache_energy_data() -> pd.DataFrame:
    """Caching main data"""
    energy_data = concurrent_buildings()
    energy_data["Community Area"] = (
        energy_data["Community Area"].astype(str).str.strip().str.title()
    )
    return energy_data


@st.cache_data
def cache_geojson(tolerance: float = 0.00259) -> dict:
    """Caching geojson data.

    Default tolerance is 0.00259 from balancing from appearence and rendering time
    """
    geojson_data = load_neighborhood_geojson(
        "/project/src/data/chicago_geo/neighborhood_chi.geojson"
    )
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

    # Simplify geometry in memory
    gdf["geometry"] = gdf["geometry"].simplify(
        tolerance=tolerance, preserve_topology=True
    )

    # Convert back to dict if needed downstream
    return json.loads(gdf.to_json())


@st.cache_data
def metric_list() -> list:
    """Loading list of metrics used for project"""
    return [
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


@st.cache_data
def year_lists() -> list:
    """List of all years"""
    energy_data = cache_energy_data()
    years_list = sorted(
        [int(year) for year in sorted(energy_data["Data Year"].dropna().unique())]
    )
    full_year_list = ["Average (All Years)"] + years_list
    return years_list, full_year_list


# Graph specific Dataframes #-------------------------------------------------------------------


@st.cache_data
def cache_build_all_aggregates(
    df: pd.DataFrame, metrics: list[str]
) -> dict[str, pd.DataFrame]:
    """Cache aggregated metrics for all variables"""
    return {metric: aggregate_metric(df, metric) for metric in metrics}


@st.cache_data
def cache_build_all_year_charts(
    agg_data: dict[str, pd.DataFrame], geojson: dict
) -> dict[str, alt.Chart]:
    """Cache Altair charts using pre-aggregated data"""
    charts = {}
    for metric, agg in agg_data.items():
        chart = plot_choropleth(geojson, agg, metric, year=None)
        charts[metric] = chart
    return charts


# Graph Helper Functions #-------------------------------------------------------------------


def style_matplotlib(fig: Figure, ax: Axes = None) -> None:
    """Apply consistent dark theme styling to Matplotlib figures."""
    # Figure and axes background
    fig.patch.set_facecolor("#0E1117")
    if ax is not None:
        ax.set_facecolor("#0E1117")

        # Tick labels and axis labels
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

        # Spines (border)
        for spine in ax.spines.values():
            spine.set_color("white")


def render_yearly_map(
    year: int, geojson_data: json, data: pd.DataFrame, log_scale: bool = False
) -> alt.Chart:
    """To modularize map rendering"""
    chart = plot_building_count_map(geojson_data, data, year=year)
    base, overlay = chart.layer
    overlay = overlay.encode(
        color=alt.Color(
            "Building_Count:Q",
            scale=alt.Scale(
                type="log" if log_scale else "linear", domain=[10, 200], scheme="blues"
            ),
            legend=alt.Legend(title="Number of Buildings"),
        )
    )
    return alt.layer(base, overlay).properties(width=600, height=400)
