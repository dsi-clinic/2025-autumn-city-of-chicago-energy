"""Exploratroy Analysis Page"""

import time

import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_energy_data,
    cache_full_data,
    cache_geojson,
    metric_list,
    render_yearly_map,
    style_matplotlib,
    year_lists,
)
from utils.plot_utils import (
    aggregate_metric,
    plot_bar,
    plot_choropleth,
    plot_trend_by_year,
)

# -------------------- Page Setup --------------------
apply_page_config()
start = time.time()
st.title("Exploratory Dashboard")

# -------------------- Load Data --------------------
full_data = cache_full_data()
energy_data = cache_energy_data()
geojson_data = cache_geojson()
metrics_list = metric_list()
years_list, full_year_list = year_lists()

# Ensure Community Area matches pri_neigh in geojson
full_data["Community Area"] = (
    full_data["Community Area"].astype(str).str.strip().str.title()
)

energy_data["Community Area"] = (
    energy_data["Community Area"].astype(str).str.strip().str.title()
)

# ------------------- Start Dashboard --------------------

# DATA COUNT PLOTS #-------------------------------------------------------------------

# Log scale toggle
log_scale = st.checkbox("Use Log Scale", value=False)

# Initialize session state
if "playing" not in st.session_state:
    st.session_state.playing = False
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Layout columns
col1, col2 = st.columns([1, 1])

# --- Full Data Animation ---
with col1:
    st.markdown("### Full Data Building Count Over Time - Animation")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    with ctrl1:
        if st.button("▶️ Play Animation"):
            st.session_state.playing = True
    with ctrl2:
        if st.button("⏸️ Pause"):
            st.session_state.playing = False
    with ctrl3:
        selected_year = st.selectbox(
            "Current year:",
            years_list,
            index=st.session_state.current_index,
            key="year_selector",
        )
        if selected_year != years_list[st.session_state.current_index]:
            st.session_state.current_index = years_list.index(selected_year)
            st.session_state.playing = False

    animation_placeholder = st.empty()

    if st.session_state.playing:
        while st.session_state.playing and st.session_state.current_index < len(
            years_list
        ):
            current_year = years_list[st.session_state.current_index]
            with animation_placeholder.container():
                st.altair_chart(
                    render_yearly_map(current_year, geojson_data, full_data, log_scale),
                    use_container_width=True,
                )
                st.progress(st.session_state.current_index / (len(years_list) - 1))
                st.caption(f"{st.session_state.current_index + 1} of {len(years_list)}")
            time.sleep(1)
            st.session_state.current_index += 1
            if st.session_state.current_index >= len(years_list):
                st.session_state.current_index = 0
        st.session_state.playing = False
        st.rerun()
    else:
        current_year = years_list[st.session_state.current_index]
        with animation_placeholder.container():
            st.altair_chart(
                render_yearly_map(current_year, geojson_data, full_data, log_scale),
                use_container_width=True,
            )

# --- Concurrent Buildings Static Map ---
with col2:
    st.markdown("### Concurrent Buildings Count Over Time")
    cur_year = st.selectbox("Select year:", years_list, key="year_select")
    st.altair_chart(
        render_yearly_map(cur_year, geojson_data, energy_data, log_scale),
        use_container_width=True,
    )
# END OF DATA COUNT PLOTS #-------------------------------------------------------------------

# BAR CHART #-------------------------------------------------------------------

st.markdown("#### Community Area per Average Metric")

col1, col2 = st.columns(2)
with col2:
    select_neigh = st.selectbox(
        "Choose a community area:",
        energy_data["Community Area"].unique(),
        key="plot",
    )
with col1:
    select_mec = st.selectbox(
        "Choose a metric:",
        metrics_list,
        key="plot_mec",
    )

# Filter the data for the selected neighborhood
fil = energy_data[energy_data["Community Area"] == select_neigh]

# Plot the bar chart
fig10, ax10 = plot_bar(
    data=fil,
    x=select_mec,
    y="Primary Property Type",
    title=f"Average {select_mec} by Property Type in {select_neigh}",
    show_values=True,
)

for container in ax10.containers:
    ax10.bar_label(container, fmt="%.0f", fontsize=9, padding=3, color="white")

style_matplotlib(fig10, ax10)
st.pyplot(fig10)

# END OF BAR CHART #-------------------------------------------------------------------

# DUEL METRIC PLOTS #-------------------------------------------------------------------

st.markdown("### Metric Exploration")
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


# END OF DUEL MAP PLOTS #-------------------------------------------------------------------

st.divider()

# COMBO PLOTS #-------------------------------------------------------------------
col1, col2 = st.columns(2)

# --- Trend Line Plot Controls ---

st.markdown("### Line and Map Plot Combo")
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
