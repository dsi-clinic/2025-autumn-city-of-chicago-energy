"""Exploratory Analysis Page"""

import time

import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_energy_data,
    cache_full_data,
    cache_geojson,
    load_clean_energy_data_for_dashboards,
    metric_list,
    precompute_animation_maps,
    render_dashboard_section,
    render_yearly_map,
    year_lists,
)
from utils.data_utils import add_top_level_property_type

# -------------------- Page Setup --------------------
apply_page_config()
start = time.time()
st.title("Building Data Over Time")

# -------------------- Load Data --------------------
full_data = cache_full_data()
energy_data = cache_energy_data()
geojson_data = cache_geojson()
metrics_list = metric_list()
years_list, full_year_list = year_lists()

# ------------------- Start Dashboard --------------------

# DATA COUNT PLOTS #-------------------------------------------------------------------
st.divider()

# Initialize session state
if "playing" not in st.session_state:
    st.session_state.playing = False
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# -------------------- Animation Controls --------------------
st.markdown("### 🎬 Map Animation Controls")
st.markdown(
    "Watch how building counts evolve across Chicago neighborhoods over time. "
    "Use the controls below to play an animation or jump to a specific year."
)

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 6])

with ctrl_col1:
    # Play/Pause buttons with clear labels
    if not st.session_state.playing:
        if st.button("▶️  Play Animation", use_container_width=True, type="primary"):
            st.session_state.playing = True
    else:
        if st.button("⏸️  Pause", use_container_width=True):
            st.session_state.playing = False

with ctrl_col2:
    # Year selector
    selected_year = st.selectbox(
        "Jump to Year",
        years_list,
        index=st.session_state.current_index,
        key="year_selector",
        help="Select a specific year to display, or use Play to animate through all years",
    )
    if selected_year != years_list[st.session_state.current_index]:
        st.session_state.current_index = years_list.index(selected_year)
        st.session_state.playing = False

with ctrl_col3:
    # Display options
    log_scale = st.checkbox(
        "Use Logarithmic Scale",
        value=False,
        help="Apply log scale to better visualize areas with very different building counts",
    )


# Layout columns
col1, col2 = st.columns([1, 1])

# Pre-compute all maps for smooth animation (cached)
yearly_maps = precompute_animation_maps(years_list, geojson_data, full_data, log_scale)

# --- Full Data Animation ---
with col1:
    st.markdown("#### 📊 All Reporting Buildings (Animated)")
    st.caption("Buildings that reported data in each specific year")
    animation_placeholder = st.empty()

    if st.session_state.playing:
        while st.session_state.playing and st.session_state.current_index < len(
            years_list
        ):
            current_year = years_list[st.session_state.current_index]
            with animation_placeholder.container():
                st.altair_chart(
                    yearly_maps[current_year],
                    use_container_width=True,
                )
                st.progress(
                    st.session_state.current_index / (len(years_list) - 1),
                    text=f"Year {current_year} ({st.session_state.current_index + 1}/{len(years_list)})",
                )
            time.sleep(1)
            st.session_state.current_index += 1
            if st.session_state.current_index >= len(years_list):
                st.session_state.current_index = 0
                st.session_state.playing = False
    else:
        current_year = years_list[st.session_state.current_index]
        with animation_placeholder.container():
            st.altair_chart(
                yearly_maps[current_year],
                use_container_width=True,
            )

# --- Complete-Data Buildings Static Map ---
with col2:
    st.markdown("#### 🏢 Consistent Reporters (Static)")
    st.caption("Buildings that reported every year, 2016-2023")
    st.altair_chart(
        render_yearly_map(None, geojson_data, energy_data, log_scale),
        use_container_width=True,
    )
# END OF DATA COUNT PLOTS #-------------------------------------------------------------------

st.divider()

#  #-------------------------------------------------------------------

st.markdown("### 📈 Metric Explorer")
st.markdown(
    "Choose an energy metric and building category below to see how performance varies "
    "across Chicago neighborhoods and building types."
)
st.markdown("")

explorer_data = load_clean_energy_data_for_dashboards(
    restrict_to_concurrent=True, concurrent_start=2016, concurrent_end=2023
)
explorer_data = add_top_level_property_type(explorer_data)

render_dashboard_section(
    metric_list=metrics_list,
    energy_data=explorer_data,
    geojson_data=cache_geojson(),
    key_prefix="Variable",
)
