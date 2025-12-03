"""Page for comparing the data"""

import time

import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_energy_data,
    cache_full_data,
    cache_geojson,
    metric_list,
    style_matplotlib,
    year_lists,
)
from utils.plot_utils import (
    aggregate_metric,
    plot_choropleth,
    plot_trend_by_year,
)

# -------------------- Page Setup --------------------
apply_page_config()
start = time.time()
st.title("Comparison Dashboard")

# -------------------- Load Data --------------------
full_data = cache_full_data()
energy_data = cache_energy_data()
geojson_data = cache_geojson()
metrics_list = metric_list()
years_list, full_year_list = year_lists()

eng_score_year = 2018

# ------------------- Start Dashboard --------------------

# DUEL METRIC PLOTS #-------------------------------------------------------------------

st.markdown("#### Duel Metric Trends")

col1, col2 = st.columns(2)

with col1:
    selected1 = st.selectbox(
        "Choose first metric:", metrics_list, key="trend_top_first"
    )
    fig1, ax1 = plot_trend_by_year(energy_data, [selected1], "mean")[0]
    style_matplotlib(fig1, ax1)
    st.pyplot(fig1)


with col2:
    default_index = (metrics_list.index(selected1) + 1) % len(metrics_list)
    selected2 = st.selectbox(
        "Choose second metric:",
        metrics_list,
        index=default_index,
        key="trend_top_second",
    )
    fig2, ax2 = plot_trend_by_year(energy_data, [selected2], "mean")[0]
    style_matplotlib(fig2, ax2)
    st.pyplot(fig2)


# END OF DUEL METRIC PLOTS #-------------------------------------------------------------------

# DUEL MAP PLOTS #-------------------------------------------------------------------
st.markdown("#### Duel Metric Map")

col1, col2 = st.columns(2)

with col1:
    dual_map_metric_one = st.selectbox(
        "Choose first metric:", metrics_list, key="dual_map_one"
    )
    agg_df = aggregate_metric(energy_data, dual_map_metric_one)
    fig1 = plot_choropleth(geojson_data, agg_df, dual_map_metric_one)
    st.altair_chart(fig1, use_container_width=True)

with col2:
    default_index = (metrics_list.index(selected1) + 1) % len(metrics_list)
    dual_map_metric_two = st.selectbox(
        "Choose second metric:",
        metrics_list,
        index=default_index,
        key="dual_map_two",
    )
    agg_df = aggregate_metric(energy_data, dual_map_metric_two)
    fig2 = plot_choropleth(geojson_data, agg_df, dual_map_metric_two)
    st.altair_chart(fig2, use_container_width=True)

# END OF DUEL MAP PLOTS #-------------------------------------------------------------------

st.divider()

# ENERGY STAR MAP #-------------------------------------------------------------------
st.markdown("### Energy Score Exploration")

map_filtered_df = energy_data.copy()

trend_year = st.selectbox("Year for Map", full_year_list, key="energy_year")

if trend_year != "Average (All Years)":
    if int(trend_year) < eng_score_year:
        st.markdown("##### ***Before Chicago Energy Rating***")

if trend_year == "Average (All Years)":
    map_year_arg = None
else:
    map_year_arg = trend_year
    map_filtered_df = map_filtered_df[map_filtered_df["Data Year"] == map_year_arg]

col1, col2 = st.columns(2)
with col1:
    agg_df = aggregate_metric(map_filtered_df, "ENERGY STAR Score")

    eng_map = plot_choropleth(
        geojson_data, agg_df, "ENERGY STAR Score", year=map_year_arg
    )
    st.altair_chart(eng_map, use_container_width=True)

with col2:
    agg_df = aggregate_metric(map_filtered_df, "Chicago Energy Rating")

    eng_map = plot_choropleth(
        geojson_data, agg_df, "Chicago Energy Rating", year=map_year_arg
    )
    st.altair_chart(eng_map, use_container_width=True)

# END of ENERGY STAR MAP #-------------------------------------------------------------------

# COMBO PLOTS #-------------------------------------------------------------------
col1, col2 = st.columns(2)

# --- Trend Line Plot Controls ---

st.markdown("### Line and Map Plot Combo for Metrics")
trend_row1 = st.columns(2)
trend_row2 = st.columns(2)

with trend_row1[0]:
    trend_metric = st.selectbox("Metric Selection", metrics_list, key="combo_metric")
with trend_row1[1]:
    valid_trend_neighborhoods = [
        n
        for n in sorted(energy_data["Community Area"].dropna().unique())
        if energy_data[energy_data["Community Area"] == n][trend_metric]
        .dropna()
        .shape[0]
        > 0
    ]
    invalid_trend_neighborhoods = sorted(
        set(energy_data["Community Area"].dropna()) - set(valid_trend_neighborhoods)
    )
    trend_neighborhood = st.selectbox(
        "Community Area Selection", ["All"] + valid_trend_neighborhoods
    )

with trend_row2[0]:
    valid_trend_building_types = [
        b
        for b in sorted(energy_data["Primary Property Type"].dropna().unique())
        if energy_data[energy_data["Primary Property Type"] == b][trend_metric]
        .dropna()
        .shape[0]
        > 0
    ]
    invalid_trend_building_types = sorted(
        set(energy_data["Primary Property Type"].dropna())
        - set(valid_trend_building_types)
    )
    trend_building_type = st.selectbox(
        "Building Type Selection", ["All"] + valid_trend_building_types
    )
with trend_row2[1]:
    trend_year = st.selectbox("Trend Year for Map", full_year_list, key="combo_year")

col1, col2 = st.columns(2)

# --- Trend Line Plot ---
with col1:
    st.markdown("#### Metric Over Time Line Plot")
    trend_filtered_df = energy_data.copy()
    if trend_neighborhood != "All":
        trend_filtered_df = trend_filtered_df[
            trend_filtered_df["Community Area"] == trend_neighborhood
        ]
    if trend_building_type != "All":
        trend_filtered_df = trend_filtered_df[
            trend_filtered_df["Primary Property Type"] == trend_building_type
        ]

    fig1, ax1 = plot_trend_by_year(trend_filtered_df, [trend_metric], agg="mean")[0]
    title_parts = [f"{trend_metric} Over Time"]
    if trend_neighborhood != "All":
        title_parts.append(f"in {trend_neighborhood}")
    if trend_building_type != "All":
        title_parts.append(f"for {trend_building_type}")
    ax1.set_title(" • ".join(title_parts), fontsize=14)

    fig1.patch.set_facecolor("#0E1117")
    ax1.set_facecolor("#0E1117")
    ax1.tick_params(colors="white")
    ax1.title.set_color("white")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    for spine in ax1.spines.values():
        spine.set_color("white")

    st.pyplot(fig1)

# --- Choropleth Map ---
with col2:
    st.markdown("#### Choropleth Map")
    map_filtered_df = energy_data.copy()
    if trend_neighborhood != "All":
        map_filtered_df = map_filtered_df[
            map_filtered_df["Community Area"] == trend_neighborhood
        ]
    if trend_building_type != "All":
        map_filtered_df = map_filtered_df[
            map_filtered_df["Primary Property Type"] == trend_building_type
        ]
    if trend_year != "Average (All Years)":
        map_filtered_df = map_filtered_df[
            map_filtered_df["Data Year"] == int(trend_year)
        ]

    map_year_arg = None if trend_year == "Average (All Years)" else int(trend_year)
    agg_df = aggregate_metric(map_filtered_df, trend_metric)
    map_chart = plot_choropleth(geojson_data, agg_df, trend_metric, year=map_year_arg)
    st.altair_chart(map_chart, use_container_width=True)
# COMBO GRAPH END #-------------------------------------------------------------------
