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
    add_compliance_status,
    add_top_level_property_type,
    assign_effective_year_built,
    categorize_time_built,
    clean_property_type,
    clean_year_built,
    concurrent_buildings,
    covered_assign_top_types,
    load_community_geojson,
    load_covered_buildings,
    load_data,
    load_neighborhood_geojson,
    merge_covered_with_benchmarking_new,
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
def cache_covered_buildings() -> pd.DataFrame:
    """Load and cache the Covered Buildings dataset for dashboard use.

    - Loads raw covered buildings data
    - Standardizes Community Area formatting
    - Assigns Top Level Property Type
    """
    covered_df = load_covered_buildings()
    covered_df = covered_assign_top_types(covered_df)

    return covered_df


@st.cache_resource
def cache_geojson(tolerance: float = 0.00259) -> dict:
    """Caching geojson data.

    Default tolerance is 0.00259 from balancing from appearence and rendering time.
    Uses cache_resource because geojson is a large, immutable read-only asset —
    avoids the pickle/unpickle overhead of cache_data on every cache hit.
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


@st.cache_resource
def cache_community_geojson(tolerance: float = 0.00259) -> dict:
    """Cache community area geojson of Chicago.

    Uses cache_resource because geojson is a large, immutable read-only asset —
    avoids the pickle/unpickle overhead of cache_data on every cache hit.
    """
    geojson_data = load_community_geojson()
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

    gdf["geometry"] = gdf["geometry"].simplify(
        tolerance=tolerance, preserve_topology=True
    )

    return json.loads(gdf.to_json())


@st.cache_data(show_spinner=False)
def cache_full_data_prepped(
    energy_cols: list[str],
    reporting_status_col: str = "Reporting Status",
    default_year: int | None = 2018,
) -> tuple[pd.DataFrame, list[int]]:
    """Load + clean + merge + derive columns needed for dashboard pages."""
    energy_df = cache_full_data()
    energy_df = clean_property_type(energy_df)
    energy_df = clean_year_built(energy_df)
    energy_df = assign_effective_year_built(energy_df)
    energy_df = add_top_level_property_type(benchmark_df=energy_df)

    covered_df = cache_covered_buildings()
    covered_df = covered_assign_top_types(covered_df)

    full_data = merge_covered_with_benchmarking_new(
        covered_df=covered_df,
        benchmark_df=energy_df,
        verbose=False,
    )

    full_data = add_compliance_status(
        full_data,
        energy_cols=energy_cols,
        reporting_status_col=reporting_status_col,
        inplace=False,
    )

    full_data = categorize_time_built(full_data)

    years_list = sorted(
        [int(y) for y in full_data["Data Year"].dropna().unique().tolist()]
    )

    _ = default_year  # kept for future use if you want to enforce ordering

    return full_data, years_list


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
def cache_aggregate_metric(dff: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Cached wrapper around :func:`aggregate_metric`.

    The underlying groupby is small but recomputed on every Streamlit rerun
    (i.e. every widget interaction). Caching the result keyed on (df, metric)
    avoids redoing the groupby when the user only changes an unrelated control.
    """
    return aggregate_metric(dff, metric)


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


@st.cache_data
def precompute_animation_maps(
    years: list[int], geojson_data: dict, data: pd.DataFrame, log_scale: bool = False
) -> dict[int, alt.Chart]:
    """Pre-compute all yearly maps for smooth animation without page reloads.

    This function caches the expensive map generation, enabling smooth animation
    by only updating the container content rather than triggering full page reloads.
    """
    return {
        year: render_yearly_map(year, geojson_data, data, log_scale) for year in years
    }


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
    # Filters ---------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<p style="color:#1e3a5f;font-weight:600;font-size:1rem;margin:0 0 0.5rem 0;">Filter & Metric Selection</p>',
            unsafe_allow_html=True,
        )
        _, sel_time_built, sel_ppt, sel_tlpt, sel_ca = build_standard_filters(
            energy_data,
            include_top_level=True,
            page_prefix=key_prefix,
            include_category_selector=False,
        )

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            trend_year = st.selectbox(
                "Trend Year for Map", full_year_list, key=f"{key_prefix}_year"
            )
        with fcol2:
            metric = st.selectbox(
                f"Choose {key_prefix}:", metric_list, key=f"{key_prefix}_metric"
            )

    # Apply filters ---------------------------------------------------------
    filtered_df = filter_energy_by_selections(
        energy_data,
        sel_time_built=sel_time_built,
        sel_ppt=sel_ppt,
        sel_ca=sel_ca,
        sel_tlpt=sel_tlpt,
    )

    if filtered_df.empty:
        st.warning(
            "No buildings match the selected filters. Please broaden your selections."
        )
        st.stop()

    map_filtered = filtered_df.copy()
    if trend_year != "Average (All Years)":
        map_filtered = map_filtered[map_filtered["Data Year"] == int(trend_year)]

    com_df = filtered_df[filtered_df[metric].notna()]

    # Graphs ---------------------------------------------------------------
    col1, col2 = st.columns(2)

    # Create three columns: left content, spacer, right content
    col1, spacer, col2 = st.columns([1, 0.1, 0.9])

    with col1:
        # Map
        map_year_arg = None if trend_year == "Average (All Years)" else int(trend_year)
        agg_df = cache_aggregate_metric(map_filtered, metric)
        map_chart = plot_choropleth(
            geojson_data, agg_df, metric, year=map_year_arg
        ).properties(height=500)
        st.altair_chart(map_chart, use_container_width=True)
        st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

        # Trend Line Plot
        st.markdown(f"##### Trend over time of {metric}")
        fig2, ax2 = plot_trend_by_year(com_df, [metric], "mean")[0]
        ax2.set_title("")

        style_matplotlib(fig2, ax2)
        st.pyplot(fig2)

    with col2:
        # Bar Chart
        st.markdown(f"##### Average {metric} by Property Type")
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
    sel_time_built: list[str] | str,
    sel_ppt: list[str] | str,
    sel_ca: list[str] | str,
    sel_tlpt: list[str] | str | None = None,
) -> pd.DataFrame:
    """Filter the energy dataframe by standard selection lists or single values.

    Parameters
    ----------
    energy_df :
        Input dataframe to filter.
    sel_time_built :
        Selected Time Built categories (list or single string).
        If "All", includes all values.
    sel_ppt :
        Selected Primary Property Type values (list or single string).
        If "All", includes all values.
    sel_ca :
        Selected Community Area values (list or single string).
        If "All", includes all values.
    sel_tlpt :
        Selected Top Level Property Type values (list or single string),
        or None to skip this filter. If "All", includes all values.

    Returns:
    -------
    pd.DataFrame
        Filtered dataframe respecting all non‑None selections.
    """

    def normalize_selection(sel: list[str] | str, column_name: str) -> list[str]:
        """Convert string to list and handle 'All' special case."""
        if sel == "All":
            return energy_df[column_name].dropna().unique().tolist()
        if isinstance(sel, str):
            return [sel]
        return sel

    # Normalize all selections to lists
    time_built_list = normalize_selection(sel_time_built, "Time Built")
    ppt_list = normalize_selection(sel_ppt, "Primary Property Type")
    ca_list = normalize_selection(sel_ca, "Community Area")

    mask = (
        energy_df["Time Built"].isin(time_built_list)
        & energy_df["Primary Property Type"].isin(ppt_list)
        & energy_df["Community Area"].isin(ca_list)
    )

    if sel_tlpt is not None:
        tlpt_list = normalize_selection(sel_tlpt, "Top Level Property Type")
        mask &= energy_df["Top Level Property Type"].isin(tlpt_list)

    return energy_df[mask]


def build_standard_filters(
    energy_df: pd.DataFrame,
    include_top_level: bool = True,
    page_prefix: str = "filters",
    include_category_selector: bool = True,
) -> tuple[str | None, list[str], list[str], list[str], list[str]]:
    """Create standard classification + multiselect filters in Streamlit.

    When ``include_category_selector`` is False, the category dropdown is
    omitted and None is returned as the first element of the tuple. Use this
    when the calling page already exposes a category selector elsewhere.
    """
    # Display-friendly labels for category options
    _category_display = {
        "Top Level Property Type": "Top-Level Type",
        "Primary Property Type": "Sub-Type",
        "Time Built": "Time Built",
        "Community Area": "Community Area",
    }

    if include_category_selector:
        category_options = [
            "Top Level Property Type",
            "Primary Property Type",
            "Time Built",
            "Community Area",
        ]
        category_col = st.selectbox(
            "Classify buildings by",
            options=category_options,
            index=category_options.index("Time Built"),
            format_func=lambda c: _category_display.get(c, c),
            key=f"{page_prefix}_category_select",
        )
    else:
        category_col = None

    col1, col2, col3, col4 = st.columns(4)

    if include_top_level:
        with col1:
            tlpt_opts = sorted(
                energy_df["Top Level Property Type"].dropna().unique().tolist()
            )
            sel_tlpt = st.multiselect(
                "Top-Level Type",
                options=tlpt_opts,
                default=tlpt_opts,
                key=f"{page_prefix}_tlpt",
                help="Broad building use category (e.g. Commercial, Residential)",
            )
        ppt_col = col2
    else:
        sel_tlpt = []
        ppt_col = col1

    with ppt_col:
        ppt_opts = sorted(energy_df["Primary Property Type"].dropna().unique().tolist())
        sel_ppt = st.multiselect(
            "Sub-Type",
            options=ppt_opts,
            default=ppt_opts,
            key=f"{page_prefix}_ppt",
            help="Detailed property type (e.g. Office, Hospital, K-12 School)",
        )

    with col3:
        time_built_opts = sorted(energy_df["Time Built"].dropna().unique().tolist())
        sel_time_built = st.multiselect(
            "Time Built",
            options=time_built_opts,
            default=time_built_opts,
            key=f"{page_prefix}_time_built",
            help="Era when the building was constructed",
        )

    with col4:
        ca_opts = sorted(energy_df["Community Area"].dropna().unique().tolist())
        sel_ca = st.multiselect(
            "Community Area",
            options=ca_opts,
            default=ca_opts,
            key=f"{page_prefix}_community_area",
            help="Chicago community area where the building is located",
        )

    return category_col, sel_time_built, sel_ppt, sel_tlpt, sel_ca


def show_helpful_filter_error(
    filtered_df: pd.DataFrame,
    original_df: pd.DataFrame,
    filter_selections: dict[str, any],
) -> None:
    """Display actionable error message when filters produce no results.

    Args:
        filtered_df: The filtered DataFrame (empty)
        original_df: The original unfiltered DataFrame
        filter_selections: Dict mapping filter names to selected values
            Example: {
                "Time Built": ["Pre-1945", "1945-1969"],
                "Primary Property Type": ["Office"],
                "Community Area": ["Loop"],
            }
    """
    st.error("### ⚠️ No buildings match your current filter selections")

    st.markdown(
        "Your filters are too restrictive. Try broadening your selections to see data."
    )

    # Show current filter settings in an expander
    with st.expander("📋 Current Filter Settings", expanded=True):
        for filter_name, selected_values in filter_selections.items():
            if isinstance(selected_values, list):
                if len(selected_values) == 0:
                    st.markdown(f"- **{filter_name}**: ❌ None selected")
                else:
                    st.markdown(
                        f"- **{filter_name}**: {', '.join(map(str, selected_values))}"
                    )
            else:
                st.markdown(f"- **{filter_name}**: {selected_values}")

    # Identify which filter is most restrictive
    st.markdown("### 💡 Suggestions:")

    filter_match_counts = {}
    for filter_name, selected_values in filter_selections.items():
        if isinstance(selected_values, list) and len(selected_values) > 0:
            # For multi-select filters
            matching_count = original_df[
                original_df[filter_name].isin(selected_values)
            ].shape[0]
            filter_match_counts[filter_name] = matching_count
        elif not isinstance(selected_values, list) and selected_values != "All":
            # For single-select filters
            matching_count = original_df[
                original_df[filter_name] == selected_values
            ].shape[0]
            filter_match_counts[filter_name] = matching_count

    if filter_match_counts:
        most_restrictive = min(filter_match_counts, key=filter_match_counts.get)
        st.markdown(
            f"- **Most restrictive filter**: `{most_restrictive}` "
            f"(only {filter_match_counts[most_restrictive]:,} buildings match)"
        )
        st.markdown(
            f"- **Recommendation**: Try selecting more values for `{most_restrictive}`"
        )

    # Show quick actions
    st.markdown("### 🔧 Quick Actions:")
    st.markdown(
        "1. Use the **filter selectors above** to broaden your choices\n"
        "2. Select **'All'** for filters you don't need\n"
        "3. Try a **different combination** of filters"
    )


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
            "% of Buildings That Reported",
            "% of Buildings That Did Not Report",
            "Number of Buildings That Reported",
            "Number of Buildings That Did Not Report",
        ],
        index=0,
    )

    if metric_option == "% of Buildings That Reported":
        return "share_submitted", "% of Buildings That Reported"
    if metric_option == "% of Buildings That Did Not Report":
        return "share_non_compliant", "% of Buildings That Did Not Report"
    if metric_option == "Number of Buildings That Reported":
        return "n_submitted", "Number of Buildings That Reported"
    return "n_non_compliant", "Number of Buildings That Did Not Report"


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
