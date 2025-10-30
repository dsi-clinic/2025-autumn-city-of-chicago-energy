"""Intial Explore Page for dashboard"""

import importlib
import sys
import warnings
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import streamlit as st

from utils.data_utils import concurrent_buildings, load_data
from utils.plot_utils import plot_trend_by_year

# --- Local imports ---
sys.path.append("..")
import utils.mira_wk2_mapping.mira_mapping as mm
from utils.mira_wk2_mapping.data_loader import (
    load_energy_datasets,
    load_neighborhood_geojson,
)

# --- Configurations ---
warnings.filterwarnings("ignore", category=FutureWarning, module="altair")
importlib.reload(mm)
alt.data_transformers.disable_max_rows()

ROOT = Path("..")
DATA_DIR = ROOT / "data/chicago_energy_benchmarking"
GEO_PATH = ROOT / "data/chicago_geo/neighborhood_chi.geojson"

combined_df = load_energy_datasets(DATA_DIR)
neighborhood_geo = load_neighborhood_geojson(GEO_PATH)

alt.data_transformers.disable_max_rows()

sys.path.append(str(Path.cwd().parent))
sys.path.append("..")

ROOT = Path("..")
DATA = ROOT / "data/chicago_energy_benchmarking"
GEOJSON = ROOT / "data/chicago_geo/neighborhood_chi.geojson"

dff = load_energy_datasets(DATA)
geo = load_neighborhood_geojson(GEOJSON)

importlib.reload(mm)

st.title("Exploratory Dashboard")

energy_data = concurrent_buildings(load_data())

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


col1, col2 = st.columns(2)

with col1:
    selected1 = st.selectbox("Choose first metric:", variables[1:])
    plot_trend_by_year(energy_data, [selected1], "mean")
    st.pyplot(plt)

with col2:
    # Set default index to the next available option, but keep all options
    default_index = (variables[1:].index(selected1) + 1) % len(variables[1:])
    selected2 = st.selectbox(
        "Choose second metric:", variables[1:], index=default_index
    )
    plot_trend_by_year(energy_data, [selected2], "mean")
    st.pyplot(plt)

st.divider()

metrics = [
    "Site EUI (kBtu/sq ft)",
    "Source EUI (kBtu/sq ft)",
    "Weather Normalized Site EUI (kBtu/sq ft)",
    "Weather Normalized Source EUI (kBtu/sq ft)",
    "Total GHG Emissions (Metric Tons CO2e)",
    "GHG Intensity (kg CO2e/sq ft)",
    "Water Use (kGal)",
    "ENERGY STAR Score",
]

st.subheader("Maps")


selected_metric = st.selectbox("Select a metric to display:", metrics, index=1)
st.write("Selected metric:", selected_metric)

map_chart = mm.plot_metric_change_map(neighborhood_geo, combined_df, [selected_metric])
st.altair_chart(map_chart[0], use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    selected1 = st.selectbox("Choose first metric:", metrics, key="metric1")
    map_chart1 = mm.plot_metric_change_map(neighborhood_geo, combined_df, [selected1])
    st.altair_chart(map_chart1[0], use_container_width=True)

with col2:
    selected2 = st.selectbox("Choose second metric:", metrics, index=1, key="metric2")
    map_chart2 = mm.plot_metric_change_map(neighborhood_geo, combined_df, [selected2])
    st.altair_chart(map_chart2[0], use_container_width=True)


st.divider()

# col1, col2 = st.columns(2)

# with col1:
#     selected1 = st.selectbox("Choose first metric:", metrics)
#     chart1 = mm.plot_metric_change_map(neighborhood_geo, combined_df, selected1)
#     st.altair_chart(chart1, use_container_width=True)

# with col2:
#     # Default to a different metric, but allow re-selection
#     default_index = (metrics.index(selected1) + 1) % len(metrics)
#     selected2 = st.selectbox("Choose second metric:", metrics, index=default_index)
#     chart2 = mm.plot_metric_change_map(neighborhood_geo, combined_df, selected2)
#     st.altair_chart(chart2, use_container_width=True)
