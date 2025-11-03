"""Statistics Page for dashboard"""

import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_energy_data,
)
from utils.data_utils import pivot_energy_metric
from utils.plot_utils import plot_building_energy_deltas
from utils.stats_utils import (
    generate_descriptive_stats_by_year,
)

apply_page_config()

energy_data = cache_energy_data()

st.dataframe(generate_descriptive_stats_by_year(energy_data))


selected_metric = "Site EUI (kBtu/sq ft)"

pivot_energy = pivot_energy_metric(
    metric_col=selected_metric, input_df=energy_data, start_year=2016, end_year=2023
)

fig10, axes10 = plot_building_energy_deltas(
    pivot_df=pivot_energy, metric_name=selected_metric, start_year=2016, end_year=2023
)

fig10.patch.set_facecolor("#0E1117")
for ax in axes10:
    ax.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.set_title(ax.get_title(), color="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")

if fig10._suptitle:
    fig10._suptitle.set_color("white")

st.pyplot(fig10)
