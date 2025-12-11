"""Exploratory Analysis Page"""

import time

import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_energy_data,
    cache_full_data,
    cache_geojson,
    metric_list,
    render_dashboard_section,
    render_yearly_map,
    year_lists,
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

# ------------------- Start Dashboard --------------------

# DATA COUNT PLOTS #-------------------------------------------------------------------
st.divider()

# Initialize session state
if "playing" not in st.session_state:
    st.session_state.playing = False
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Controls
ctrl1, _, ctrl2, _, ctrl3, _, ctrl4, _ = st.columns([1, 0.1, 1, 0.1, 1, 0.1, 1, 4.5])
with ctrl1:
    log_scale = st.checkbox("Use Log Scale", value=False)
with ctrl2:
    selected_year = st.selectbox(
        "Select Year:",
        years_list,
        index=st.session_state.current_index,
        key="year_selector",
    )
    if selected_year != years_list[st.session_state.current_index]:
        st.session_state.current_index = years_list.index(selected_year)
        st.session_state.playing = False
with ctrl3:
    if st.button("▶️ Play Animation"):
        st.session_state.playing = True
with ctrl4:
    if st.button("⏸️ Pause Animation"):
        st.session_state.playing = False


# Layout columns
col1, col2 = st.columns([1, 1])

# --- Full Data Animation ---
with col1:
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
    st.altair_chart(
        render_yearly_map(None, geojson_data, energy_data, log_scale),
        use_container_width=True,
    )
# END OF DATA COUNT PLOTS #-------------------------------------------------------------------

st.divider()

#  #-------------------------------------------------------------------

render_dashboard_section(
    metric_list=metrics_list,
    geojson_data=cache_geojson(),
    key_prefix="Variable",
)
