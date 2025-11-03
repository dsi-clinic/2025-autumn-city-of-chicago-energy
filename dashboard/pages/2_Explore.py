"""Initial Explore Page for dashboard"""

import time

import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_build_all_aggregates,
    cache_build_all_year_charts,
    cache_energy_data,
    cache_geojson,
)
from utils.plot_utils import (
    plot_bar,
    plot_building_count_map,
    plot_choropleth,
    plot_trend_by_year,
)

# -------------------- Page Setup --------------------
apply_page_config()
start = time.time()
st.title("Exploratory Dashboard")

# -------------------- Load Data --------------------
energy_data = cache_energy_data()
geojson_data = cache_geojson()

variables = [
    "ENERGY STAR Score",
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

# # -------------------- Precompute All-Year Charts --------------------
agg_energy_data = cache_build_all_aggregates(energy_data, variables)
all_year_charts = cache_build_all_year_charts(agg_energy_data, geojson_data)

# -------------------- Trend Plots --------------------
col1, col2 = st.columns(2)

with col1:
    selected1 = st.selectbox(
        "Choose first metric:", variables[1:], key="trend_top_first"
    )
    fig1, ax1 = plot_trend_by_year(energy_data, [selected1], "mean")

    # Styling: black background, white ticks
    fig1.patch.set_facecolor("#0E1117")  # Figure background
    ax1.set_facecolor("#0E1117")  # Axes background
    ax1.tick_params(colors="white")  # Tick label color
    ax1.title.set_color("white")  # Title color
    ax1.xaxis.label.set_color("white")  # X-axis label color
    ax1.yaxis.label.set_color("white")  # Y-axis label color
    for spine in ax1.spines.values():  # Optional: white border
        spine.set_color("white")

    st.pyplot(fig1)

with col2:
    default_index = (variables[1:].index(selected1) + 1) % len(variables[1:])
    selected2 = st.selectbox(
        "Choose second metric:",
        variables[1:],
        index=default_index,
        key="trend_top_second",
    )
    fig2, ax2 = plot_trend_by_year(energy_data, [selected2], "mean")

    # Styling: black background, white ticks
    fig2.patch.set_facecolor("#0E1117")
    ax2.set_facecolor("#0E1117")
    ax2.tick_params(colors="white")
    ax2.title.set_color("white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    for spine in ax2.spines.values():
        spine.set_color("white")

    st.pyplot(fig2)


fig10, ax10 = plot_bar(
    data=energy_data,
    x="Primary Property Type",
    y="Site EUI (kBtu/sq ft)",
    title="Average Site EUI by Neighborhood",
    rotate_xticks=90,
    show_values=True,
)

fig10.patch.set_facecolor("#0E1117")
ax10.set_facecolor("#0E1117")
ax10.tick_params(colors="white")
ax10.title.set_color("white")
ax10.xaxis.label.set_color("white")
ax10.yaxis.label.set_color("white")
for spine in ax10.spines.values():
    spine.set_color("white")

for container in ax10.containers:
    ax10.bar_label(container, fmt="%.0f", fontsize=9, padding=3, color="white")

st.pyplot(fig10)


st.divider()
# -------------------- Map Selection --------------------
st.subheader("Maps")

available_years = sorted(energy_data["Data Year"].dropna().unique())
year_options = ["Average (All Years)"] + sorted(
    [int(year) for year in available_years], reverse=True
)

with st.container():
    VarCol, BuildCol = st.columns(2)

    with VarCol:
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                map_select = st.selectbox(
                    "Choose metric for map:", variables, key="map_metric"
                )
            with col2:
                year_select = st.selectbox("Choose year:", year_options, key="year1")
                year_arg = (
                    None if year_select == "Average (All Years)" else int(year_select)
                )

        # Use cached chart if year is None, otherwise recompute chart using cached aggregation
        if year_arg is None:
            map_chart = all_year_charts[map_select]
        else:
            agg = agg_energy_data[map_select]  # reuse cached aggregation
            map_chart = plot_choropleth(geojson_data, agg, map_select, year_arg)
            st.write("making new graph")

        st.altair_chart(map_chart, width="stretch")

    with BuildCol:
        # st.write("nothing")
        energy_data["Community Area"] = (
            energy_data["Community Area"].astype(str).str.strip().str.title()
        )
        map_build = plot_building_count_map(geojson_data, energy_data)

        st.altair_chart(map_build, width="stretch")

st.write(f"Render time: {time.time() - start:.2f} seconds")
