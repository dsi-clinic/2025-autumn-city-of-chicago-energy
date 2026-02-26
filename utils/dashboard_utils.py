"""Modularizing configurations for each page"""

import base64
import json

import altair as alt
import geopandas as gpd
import pandas as pd
import streamlit as st
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.data_utils import (
    assign_effective_year_built,
    categorize_time_built,
    clean_property_type,
    clean_year_built,
    concurrent_buildings,
    load_community_geojson,
    load_data,
    load_neighborhood_geojson,
)
from utils.plot_utils import (
    aggregate_metric,
    plot_bar,
    plot_building_count_map,
    plot_choropleth,
    plot_trend_by_year,
)

FACE_COLORS = {
    "dark": "#0E1117",
    "light": "#F1F2F6",
}

LABEL_COLORS = {
    "dark": "white",
}

SPINE_COLORS = {
    "dark": "white",
}


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


def geojson_to_data_url(geo: dict) -> str:
    """Encode a GeoJSON FeatureCollection as a data: URL (base64)."""
    payload = json.dumps(geo, separators=(",", ":")).encode("utf-8")
    b64 = base64.b64encode(payload).decode("ascii")
    return f"data:application/json;base64,{b64}"


@st.cache_data
def cache_community_geojson_url(tolerance: float = 0.00259) -> str:
    """Cached data-URL version of community-area geojson for Altair/Vega."""
    geo = cache_community_geojson(tolerance=tolerance)
    return geojson_to_data_url(geo)


@st.cache_data
def cache_community_geojson(tolerance: float = 0.00259) -> dict:
    """Cache community area geojson of Chicago."""
    geojson_data = load_community_geojson()
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

    gdf["geometry"] = gdf["geometry"].simplify(
        tolerance=tolerance, preserve_topology=True
    )

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
        "Water Use (kGal)",
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
    # get theme from streamlit
    theme = st.get_option("theme.base")
    # Set facecolors
    if theme in FACE_COLORS:
        face_color = FACE_COLORS[theme]
        fig.patch.set_facecolor(face_color)
        if ax is not None:
            ax.set_facecolor(face_color)
    # Set label colors
    if ax is not None and theme in LABEL_COLORS:
        label_color = LABEL_COLORS[theme]
        ax.tick_params(colors=label_color)
        ax.title.set_color(label_color)
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
    # Set spine colors
    if ax is not None and theme in SPINE_COLORS:
        spine_color = SPINE_COLORS[theme]
        for spine in ax.spines.values():
            spine.set_color(spine_color)


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
    return alt.layer(base, overlay).properties(height=500)


# grouped charts #-------------------------------------------------------------------s


def render_dashboard_section(
    metric_list: list,
    key_prefix: str,
    energy_data: pd.DataFrame | None = None,
    full_year_list: list | None = None,
    geojson_data: dict | None = None,
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

    # Create three columns: left content, spacer, right content
    col1, spacer, col2 = st.columns([1, 0.1, 0.9])

    with col1:
        # Map
        map_year_arg = None if trend_year == "Average (All Years)" else int(trend_year)
        agg_df = aggregate_metric(map_filtered, metric)
        map_chart = plot_choropleth(
            geojson_data, agg_df, metric, year=map_year_arg
        ).properties(height=500)
        st.altair_chart(map_chart, use_container_width=True)
        st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

        # Trend Line Plot
        st.markdown(
            f"##### Trend over time of {metric} by Year Built in {trend_neighborhood}"
        )
        fig2, ax2 = plot_trend_by_year(com_df, [metric], "mean")[0]
        ax2.set_title("")

        style_matplotlib(fig2, ax2)
        st.pyplot(fig2)

    with col2:
        # Bar Chart
        st.markdown(f"##### Average {metric} by Property Type in {trend_neighborhood}")
        com_df_b = (
            com_df
            if trend_year == "Average (All Years)"
            else com_df[com_df["Data Year"] == trend_year]
        )
        fig10, ax10 = plot_bar(
            data=com_df_b,
            x=metric,
            y="Primary Property Type",
        )
        ax10.set_ylabel("")
        fig10.set_size_inches(5, 10)
        style_matplotlib(fig10, ax10)
        fig10.subplots_adjust(top=0.94)
        st.pyplot(fig10)


def load_clean_energy_data_for_dashboards(
    min_year: int | None = None,
    max_year: int | None = None,
    restrict_to_concurrent: bool = False,
    concurrent_start: int = 2016,
    concurrent_end: int = 2023,
) -> pd.DataFrame:
    """Load and apply standard cleaning steps for all dashboard pages.

    Parameters
    ----------
    min_year :
        Minimum Data Year to keep (inclusive). If None, no lower bound filter.
    max_year :
        Maximum Data Year to keep (inclusive). If None, no upper bound filter.
    restrict_to_concurrent :
        If True, keep only buildings that have data in every year from
        `concurrent_start` to `concurrent_end`.
    concurrent_start, concurrent_end :
        Range of years to enforce when `restrict_to_concurrent` is True.

    Returns:
    -------
    pd.DataFrame
        Cleaned energy benchmarking dataframe with:
        - consistent `Primary Property Type`
        - `Time Built`
        - effective year built
        - (optionally) restricted by year and concurrency.
    """
    energy_df = load_data()
    energy_df = clean_year_built(energy_df)
    energy_df = assign_effective_year_built(energy_df)
    energy_df = clean_property_type(energy_df)
    energy_df = categorize_time_built(energy_df)

    if restrict_to_concurrent:
        energy_df = concurrent_buildings(
            energy_df, start_year=concurrent_start, end_year=concurrent_end
        )

    if min_year is not None:
        energy_df = energy_df[energy_df["Data Year"] >= min_year]
    if max_year is not None:
        energy_df = energy_df[energy_df["Data Year"] <= max_year]

    return energy_df


def filter_energy_by_selections(
    energy_df: pd.DataFrame,
    sel_time_built: list[str],
    sel_ppt: list[str],
    sel_ca: list[str],
    sel_tlpt: list[str] | None = None,
) -> pd.DataFrame:
    """Filter the energy dataframe by standard selection lists.

    Parameters
    ----------
    energy_df :
        Input dataframe to filter.
    sel_time_built :
        Selected Time Built categories.
    sel_ppt :
        Selected Primary Property Type values.
    sel_ca :
        Selected Community Area values.
    sel_tlpt :
        Selected Top Level Property Type values, or None to skip this filter.

    Returns:
    -------
    pd.DataFrame
        Filtered dataframe respecting all non‑None selections.
    """
    mask = (
        energy_df["Time Built"].isin(sel_time_built)
        & energy_df["Primary Property Type"].isin(sel_ppt)
        & energy_df["Community Area"].isin(sel_ca)
    )
    if sel_tlpt is not None:
        mask &= energy_df["Top Level Property Type"].isin(sel_tlpt)

    return energy_df[mask]


def build_standard_filters(
    energy_df: pd.DataFrame,
    include_top_level: bool = True,
    page_prefix: str = "filters",
) -> tuple[str, list[str], list[str], list[str], list[str]]:
    """Create standard classification + multiselect filters in Streamlit."""
    # Define category options for the classification dropdown
    category_options = [
        "Time Built",
        "Primary Property Type",
        "Community Area",
        "Top Level Property Type",
    ]
    category_col = st.selectbox(
        "Select category for Building Classification",
        options=category_options,
        index=category_options.index("Time Built"),
        key=f"{page_prefix}_category_select",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        time_built_opts = sorted(energy_df["Time Built"].dropna().unique().tolist())
        sel_time_built = st.multiselect(
            "Time Built",
            options=time_built_opts,
            default=time_built_opts,
            key=f"{page_prefix}_time_built",
        )

    with col2:
        ppt_opts = sorted(energy_df["Primary Property Type"].dropna().unique().tolist())
        sel_ppt = st.multiselect(
            "Primary Property Type",
            options=ppt_opts,
            default=ppt_opts,
            key=f"{page_prefix}_ppt",
        )

    if include_top_level:
        with col3:
            tlpt_opts = sorted(
                energy_df["Top Level Property Type"].dropna().unique().tolist()
            )
            sel_tlpt = st.multiselect(
                "Top Level Property Type",
                options=tlpt_opts,
                default=tlpt_opts,
                key=f"{page_prefix}_tlpt",
            )
    else:
        sel_tlpt = []

    with col4:
        ca_opts = sorted(energy_df["Community Area"].dropna().unique().tolist())
        sel_ca = st.multiselect(
            "Community Area",
            options=ca_opts,
            default=ca_opts,
            key=f"{page_prefix}_community_area",
        )

    return category_col, sel_time_built, sel_ppt, sel_tlpt, sel_ca


def aggregate_compliance_over_time(
    energy_df: pd.DataFrame,
    category_col: str,
    year_col: str = "Data Year",
    id_col: str = "ID",
) -> pd.DataFrame:
    """Aggregate compliance counts and shares by year and category."""
    group_cols = [year_col, category_col]

    agg = (
        energy_df.groupby(group_cols, dropna=False)
        .agg(
            n_buildings=(id_col, "nunique"),
            n_submitted=("SubmittedFlag", "sum"),
            n_exempt=("ExemptFlag", "sum"),
            n_not_submitted=("NotSubmittedFlag", "sum"),
            n_non_compliant=("NonCompliantFlag", "sum"),
        )
        .reset_index()
    )

    agg["share_submitted"] = agg["n_submitted"] / agg["n_buildings"].where(
        agg["n_buildings"] > 0
    )
    agg["share_non_compliant"] = agg["n_non_compliant"] / agg["n_buildings"].where(
        agg["n_buildings"] > 0
    )

    return agg


def choose_compliance_metric() -> tuple[str, str]:
    """Streamlit widget to choose compliance metric and return (value_col, y_title)."""
    metric_option = st.selectbox(
        "Compliance metric",
        options=[
            "Share submitted",
            "Share non‑compliant",
            "Number submitted",
            "Number non‑compliant",
        ],
        index=0,
    )

    if metric_option == "Share submitted":
        return "share_submitted", "Share submitted"
    if metric_option == "Share non‑compliant":
        return "share_non_compliant", "Share non‑compliant"
    if metric_option == "Number submitted":
        return "n_submitted", "Submitted buildings"
    return "n_non_compliant", "Non‑compliant buildings"


def apply_category_filter(
    agg: pd.DataFrame,
    category_col: str,
) -> tuple[pd.DataFrame, str]:
    """Streamlit selector for a single category value or 'All'."""
    class_opts = sorted(agg[category_col].dropna().unique().tolist())
    selected_class = st.selectbox(f"{category_col} filter", ["All"] + class_opts)

    if selected_class != "All":
        return agg[agg[category_col] == selected_class], selected_class
    return agg, "All"
