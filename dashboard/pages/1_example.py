"""Test page for running dashboard"""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from utils.data_utils import load_data
from utils.plot_utils import plot_trend_by_year

st.title("Simple Dashboard")

energy_df = load_data()
energy_df = energy_df[energy_df["Primary Property Type"] != "nan"]

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
    selected1 = st.selectbox("Choose first metric:", variables[1:])
    plot_trend_by_year(energy_df, [selected1], "mean")
    st.pyplot(plt)

with col2:
    # Set default index to the next available option, but keep all options
    default_index = (variables[1:].index(selected1) + 1) % len(variables[1:])
    selected2 = st.selectbox(
        "Choose second metric:", variables[1:], index=default_index
    )
    plot_trend_by_year(energy_df, [selected2], "mean")
    st.pyplot(plt)
