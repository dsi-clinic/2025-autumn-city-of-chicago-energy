"""Exploratroy Analysis Page"""

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

Scores = ["ENERGY STAR Score", "Chicago Energy Rating"]
Utility = [
    "Electricity Use (kBtu)",
    "Natural Gas Use (kBtu)",
    "District Steam Use (kBtu)",
    "District Chilled Water Use (kBtu)",
    "All Other Fuel Use (kBtu)",
    "Water Use (kGal)",
]
Energy = [
    "Site EUI (kBtu/sq ft)",
    "Source EUI (kBtu/sq ft)",
    "Weather Normalized Site EUI (kBtu/sq ft)",
    "Weather Normalized Source EUI (kBtu/sq ft)",
]
Emissions = ["Total GHG Emissions (Metric Tons CO2e)", "GHG Intensity (kg CO2e/sq ft)"]

# ------------------- Start Dashboard --------------------

# DATA COUNT PLOTS #-------------------------------------------------------------------
st.divider()
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

st.divider()

#  #-------------------------------------------------------------------

st.markdown("### Utility Graphs")
render_dashboard_section(
    metric_list=Utility,
    geojson_data=cache_geojson(),
    special_condition="All",  # resize when neighborhood == "All"
    key_prefix="Utility",
)
st.markdown("### Energy Graphs")
render_dashboard_section(
    metric_list=Energy,
    geojson_data=cache_geojson(),
    special_condition="All",  # resize when neighborhood == "All"
    key_prefix="Energy",
)
st.markdown("### Emission Graphs")
render_dashboard_section(
    metric_list=Emissions,
    geojson_data=cache_geojson(),
    special_condition="All",  # resize when neighborhood == "All"
    key_prefix="Emissions",
)
st.markdown("### Score Graphs")
render_dashboard_section(
    metric_list=Scores,
    geojson_data=cache_geojson(),
    special_condition="Chicago Energy Rating",
    key_prefix="Score",
)
