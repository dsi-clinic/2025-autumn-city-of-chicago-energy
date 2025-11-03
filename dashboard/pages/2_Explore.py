"""Intial Explore Page for dashboard"""

import matplotlib.pyplot as plt
import streamlit as st

from utils.dashboard_utils import apply_page_config
from utils.data_utils import concurrent_buildings, load_data
from utils.plot_utils import plot_trend_by_year

apply_page_config()

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
