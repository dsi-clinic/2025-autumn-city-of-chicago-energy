"""Modularizing configurations for each page"""

import json

import altair as alt
import geopandas as gpd
import pandas as pd
import streamlit as st
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.data_utils import concurrent_buildings, load_data, load_neighborhood_geojson
from utils.plot_utils import (
    aggregate_metric,
    plot_bar,
    plot_building_count_map,
    plot_choropleth,
    plot_trend_by_year,
)

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
    geojson_data = load_neighborhood_geojson()
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
        "Chicago Energy Rating",
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


# grouped charts #-------------------------------------------------------------------s


def render_dashboard_section(
    metric_list: list,
    special_condition: str,
    key_prefix: str,
    energy_data: pd.DataFrame | None = None,
    full_year_list: list | None = None,
    geojson_data: dict | None = None,
    special_resize: tuple = (8, 10),
) -> None:
    """Render a dashboard section with filters, map, bar chart, and line chart.

    Parameters
    ----------
    section_title : str
        Title for the section (e.g. "Score Graphs", "Utility Graphs").
    energy_data : pd.DataFrame
        The full dataset.
    metric_list : list
        List of metrics to choose from (e.g. Scores or Utility).
    geojson_data : dict
        GeoJSON data for choropleth plotting.
    special_condition : str
        Condition to trigger resizing (e.g. "Chicago Energy Rating" or "All").
    special_resize : tuple
        (width, height) for resizing the bar chart when condition is met.
    key_prefix : str
        Prefix for Streamlit widget keys to avoid collisions.
    """
    if energy_data is None:
        energy_data = cache_energy_data()
    if full_year_list is None:
        _, full_year_list = year_lists()
    if geojson_data is None:
        geojson_data = cache_geojson()
    # Layout rows
    trend_row1 = st.columns(2)
    trend_row2 = st.columns(2)

    # Filters ---------------------------------------------------------------
    with trend_row1[0]:
        years_build = sorted(
            [int(year) for year in energy_data["Year Built"].dropna().unique()]
        )
        year_range = st.slider(
            "Select Range of Year Built",
            min_value=min(years_build),
            max_value=max(years_build),
            value=(min(years_build), max(years_build)),
            step=1,
            key=f"{key_prefix}_slider",
        )

    with trend_row1[1]:
        trend_year = st.selectbox(
            "Trend Year for Map", full_year_list, key=f"{key_prefix}_year"
        )

    with trend_row2[0]:
        trend_building_type = st.selectbox(
            "Building Type Selection",
            ["All"] + sorted(energy_data["Primary Property Type"].dropna().unique()),
            key=f"{key_prefix}_build",
        )

    with trend_row2[1]:
        trend_neighborhood = st.selectbox(
            "Community Area Selection",
            ["All"] + sorted(energy_data["Community Area"].dropna().unique()),
            key=f"{key_prefix}_comm",
        )

    metric = st.selectbox(
        f"Choose {key_prefix}:", metric_list, key=f"{key_prefix}_metric"
    )

    # Main filter -----------------------------------------------------------
    year_built_df = energy_data[
        (energy_data["Year Built"] >= year_range[0])
        & (energy_data["Year Built"] <= year_range[1])
    ]

    map_filtered = year_built_df.copy()
    if trend_neighborhood != "All":
        map_filtered = map_filtered[
            map_filtered["Community Area"] == trend_neighborhood
        ]
    if trend_building_type != "All":
        map_filtered = map_filtered[
            map_filtered["Primary Property Type"] == trend_building_type
        ]
    if trend_year != "Average (All Years)":
        map_filtered = map_filtered[map_filtered["Data Year"] == int(trend_year)]

    # Bar filter
    if trend_neighborhood != "All":
        com_df = year_built_df[year_built_df["Community Area"] == trend_neighborhood]
    else:
        com_df = year_built_df
    com_df = com_df[com_df[metric].notna()]

    # Graphs ---------------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        map_year_arg = None if trend_year == "Average (All Years)" else int(trend_year)
        agg_df = aggregate_metric(map_filtered, metric)
        map_chart = plot_choropleth(geojson_data, agg_df, metric, year=map_year_arg)
        st.altair_chart(map_chart, use_container_width=True)

    with col2:
        com_df_b = (
            com_df
            if trend_year == "Average (All Years)"
            else com_df[com_df["Data Year"] == trend_year]
        )
        fig10, ax10 = plot_bar(
            data=com_df_b,
            x=metric,
            y="Primary Property Type",
            title=f"Average {metric} by Property Type in {trend_neighborhood}",
            show_values=True,
        )
        for container in ax10.containers:
            ax10.bar_label(container, fmt="%.0f", fontsize=9, padding=3, color="white")

        style_matplotlib(fig10, ax10)

        # Apply special resizing condition
        if key_prefix == "score":
            if metric == special_condition:
                fig10.set_size_inches(*special_resize)
        else:
            if trend_neighborhood == special_condition:
                fig10.set_size_inches(*special_resize)

        st.pyplot(fig10)

    # Line chart centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig2, ax2 = plot_trend_by_year(com_df, [metric], "mean")[0]
        style_matplotlib(fig2, ax2)
        st.pyplot(fig2)
