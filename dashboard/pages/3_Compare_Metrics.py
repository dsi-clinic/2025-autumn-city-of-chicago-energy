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
st.title("Compare Energy Metrics")

st.markdown(
    "Pick two energy metrics to compare side-by-side. See how they change over time "
    "and vary across Chicago neighborhoods."
)
st.markdown("")

# -------------------- Load Data --------------------
full_data = cache_full_data()
energy_data = cache_energy_data()
geojson_data = cache_geojson()
metrics_list = metric_list()
years_list, full_year_list = year_lists()

eng_score_year = 2018

# ------------------- Start Dashboard --------------------

st.markdown("### Select Metrics to Compare")

col1, col2 = st.columns(2)

with col1:
    selected1 = st.selectbox(
        "Choose first metric:", metrics_list, key="global_metric_1"
    )

with col2:
    default_index = (metrics_list.index(selected1) + 1) % len(metrics_list)
    selected2 = st.selectbox(
        "Choose second metric:",
        metrics_list,
        index=default_index,
        key="global_metric_2",
    )

st.divider()

# COMPARE METRIC TRENDS OVER TIME #-------------------------------------------------------------------

st.markdown("### 📊 How Have These Metrics Changed Over Time?")
st.caption("Average for all buildings, by year")
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plot_trend_by_year(energy_data, [selected1], "mean")[0]
    style_matplotlib(fig1, ax1)
    st.pyplot(fig1)


with col2:
    fig2, ax2 = plot_trend_by_year(energy_data, [selected2], "mean")[0]
    style_matplotlib(fig2, ax2)
    st.pyplot(fig2)


# END OF COMPARE METRIC TRENDS OVER TIME #-------------------------------------------------------------------

st.divider()

# COMPARE METRIC GEOGRAPHIC TRENDS #-------------------------------------------------------------------
st.markdown("### 🗺️ How Do These Metrics Vary by Neighborhood?")
st.caption("Average by Chicago community area")
st.markdown("")

map_filtered_df = energy_data.copy()

st.markdown("**Select Year for Geographic View**")
col1, col2, col3 = st.columns([2, 4, 4])
with col1:
    trend_year = st.selectbox(
        "Year:",
        full_year_list,
        key="energy_year",
        help="Choose a specific year or view average across all years"
    )

if trend_year != "Average (All Years)":
    if "Chicago Energy Rating" in [selected1, selected2]:
        if int(trend_year) < eng_score_year:
            st.markdown("##### ***Before Chicago Energy Rating***")

if trend_year == "Average (All Years)":
    map_year_arg = None
else:
    map_year_arg = trend_year
    map_filtered_df = map_filtered_df[map_filtered_df["Data Year"] == map_year_arg]

col1, col2 = st.columns(2)
with col1:
    agg_df = aggregate_metric(map_filtered_df, selected1)

    eng_map = plot_choropleth(geojson_data, agg_df, selected1, year=map_year_arg)
    st.altair_chart(eng_map, use_container_width=True)

with col2:
    agg_df = aggregate_metric(map_filtered_df, selected2)

    eng_map = plot_choropleth(geojson_data, agg_df, selected2, year=map_year_arg)
    st.altair_chart(eng_map, use_container_width=True)

# END of COMPARE METRIC GEOGRAPHIC TRENDS #-------------------------------------------------------------------
