"""Han Test Page"""

import streamlit as st

from utils.data_utils import (
    assign_effective_year_built,
    categorize_time_built,
    clean_property_type,
    concurrent_buildings,
    load_data,
    prepare_persistence,
)
from utils.plot_utils import plot_energy_persistence_chart

st.title("Energy Persistence Visualization")

energy_df = load_data()
energy_df = assign_effective_year_built(energy_df)
energy_df = concurrent_buildings(energy_df, 2016, 2023)
energy_df = clean_property_type(energy_df)
energy_df = categorize_time_built(energy_df)

df_lagged = prepare_persistence(energy_df)

# Generate the chart (replace with your plotting call and data)
fig = plot_energy_persistence_chart(
    df_lagged=df_lagged,
    property_col="Decade Built",
    id_col="ID",
    year_col="Data Year",
    delta_col="Delta",
    delta_next_col="Delta_next",
)

st.altair_chart(fig)
