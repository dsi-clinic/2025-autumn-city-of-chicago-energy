"""Modularizing configurations for each page"""

import json

import altair as alt
import geopandas as gpd
import pandas as pd
import streamlit as st

from utils.data_utils import concurrent_buildings, load_neighborhood_geojson
from utils.plot_utils import (
    aggregate_metric,
    plot_choropleth,
)


def apply_page_config() -> None:
    """Standardizing page apperance"""
    st.set_page_config(
        page_title="Chicago Energy Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def cache_energy_data() -> pd.DataFrame:
    """Caching main data"""
    return concurrent_buildings()


@st.cache_data
def cache_geojson() -> dict:
    """Caching geojson data"""
    geojson_data = load_neighborhood_geojson(
        "/project/src/data/chicago_geo/neighborhood_chi.geojson"
    )
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

    # Simplify geometry in memory
    gdf["geometry"] = gdf["geometry"].simplify(
        tolerance=0.00259, preserve_topology=True
    )

    # Convert back to dict if needed downstream
    return json.loads(gdf.to_json())


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
