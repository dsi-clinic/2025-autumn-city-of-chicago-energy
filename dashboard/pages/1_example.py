"""Test page for running dashboard"""

import logging
import time

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from utils.dashboard_utils import apply_page_config
from utils.data_utils import concurrent_buildings, load_neighborhood_geojson
from utils.plot_utils import aggregate_metric, plot_choropleth, plot_trend_by_year

apply_page_config()
start = time.time()
st.title("Simple Dashboard")

energy_df = concurrent_buildings()

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

efficiency_trends = (
    energy_df.groupby(["Data Year", "Primary Property Type"])[variables]
    .mean()
    .reset_index()
)

selected_types = [
    "multifamily housing",
    "k-12 school",
    "office",
    "hotel",
    "college/university",
    "retail store",
]

avg_efficiency = efficiency_trends[
    efficiency_trends["Primary Property Type"].isin(selected_types)
].sort_values(["Primary Property Type", "Data Year"])

top_types = energy_df["Primary Property Type"].value_counts().nlargest(6).index


sns.lineplot(
    data=efficiency_trends[efficiency_trends["Primary Property Type"].isin(top_types)],
    x="Data Year",
    y="ENERGY STAR Score",
    hue="Primary Property Type",
    marker="o",
)

plt.title("Average ENERGY STAR Score by Property Type")
plt.ylabel("Average ENERGY STAR Score")
plt.xlabel("Year")
plt.legend(title="Property Type")

st.pyplot(plt)


selected = st.selectbox("Choose metric:", variables[1:])
plot_trend_by_year(energy_df, [selected], "mean")
st.pyplot(plt)

col1, col2 = st.columns(2)

with col1:
    selected1 = st.selectbox(
        "Choose first metric:", variables[1:], key="trend_top_first"
    )
    plot_trend_by_year(energy_df, [selected1], "mean")
    st.pyplot(plt)

with col2:
    # Set default index to the next available option, but keep all options
    default_index = (variables[1:].index(selected1) + 1) % len(variables[1:])
    selected2 = st.selectbox(
        "Choose second metric:",
        variables[1:],
        index=default_index,
        key="trend_top_second",
    )
    plot_trend_by_year(energy_df, [selected2], "mean")
    st.pyplot(plt)

st.divider()

st.title("Maps")

# No loaded data inside as within concurrent_buildings() is uses load_data() within it
energy_data = concurrent_buildings()
geojson_data = load_neighborhood_geojson(
    "/project/src/data/chicago_geo/neighborhood_chi.geojson"
)

available_years = sorted(energy_data["Data Year"].dropna().unique())
year_options = ["Average (All Years)"] + sorted(
    [int(year) for year in available_years], reverse=True
)

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        map_select = st.selectbox("Choose metric:", variables)
    with col2:
        year_select = st.selectbox("Choose year:", year_options)
        year_arg = None if year_select == "Average (All Years)" else int(year_select)

agg_energy_data = {
    metric: aggregate_metric(energy_data, metric) for metric in variables
}

map = plot_choropleth(geojson_data, agg_energy_data[map_select], map_select, year_arg)
st.altair_chart(map, width="stretch")

logging.debug(f"Render time for Example: {time.time() - start:.2f} seconds")
