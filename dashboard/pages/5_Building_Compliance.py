"""Building Compliance Over Time Visualization"""

import streamlit as st

from utils.dashboard_utils import (
    aggregate_compliance_over_time,
    apply_category_filter,
    build_standard_filters,
    choose_compliance_metric,
    filter_energy_by_selections,
    load_clean_energy_data_for_dashboards,
)
from utils.data_utils import (
    add_reporting_compliance_flags,
    add_top_level_property_type,
)
from utils.plot_utils import (
    plot_compliance_rate_over_time,
    plot_compliance_status_facets,
)

st.title("Compliance with Chicago Energy Benchmarking Over Time")
st.markdown("""
This dashboard explores how buildings comply with Chicago’s energy benchmarking
requirements over time by tracking their reporting status year by year.

Use the filters to focus on specific building types, time periods, or neighborhoods
and see how often buildings report on time versus missing data or appearing only
intermittently.
""")

# --- Load & prepare data (shared helper) ---
energy_df = load_clean_energy_data_for_dashboards(
    min_year=2014,
    max_year=2023,
    restrict_to_concurrent=False,
)

# Add top‑level property types and compliance flags
energy_df = add_top_level_property_type(energy_df)
energy_df = add_reporting_compliance_flags(energy_df)

restriction_year = 2018
energy_df = energy_df[energy_df["Data Year"] >= restriction_year]

# Compliance flag relative to merged covered buildings
energy_df["Is_Compliant"] = energy_df["Reporting Status"].ne("Not present in data")

# --- Filter controls ---
category_col, sel_time_built, sel_ppt, sel_tlpt, sel_ca = build_standard_filters(
    energy_df,
    include_top_level=True,
)

# Apply common filter logic
energy_df_filtered = filter_energy_by_selections(
    energy_df,
    sel_time_built=sel_time_built,
    sel_ppt=sel_ppt,
    sel_ca=sel_ca,
    sel_tlpt=sel_tlpt,
)

if energy_df_filtered.empty:
    st.warning(
        "No buildings match the selected filters. Please broaden your selections."
    )
    st.stop()

min_years = 2
if energy_df_filtered["Data Year"].nunique() < min_years:
    st.warning(
        "Not enough years of data for the selected filters to evaluate compliance trends."
    )
    st.stop()

# --- Aggregate compliance over time (on filtered data) ---
agg = aggregate_compliance_over_time(
    energy_df=energy_df_filtered,
    category_col=category_col,
)

# --- Choose metric to plot ---
value_col, y_title = choose_compliance_metric()

# --- Filter single category vs All ---
agg_plot, selected_class = apply_category_filter(agg, category_col)

# --- Plot 1: compliance over time (line chart) ---
st.subheader("Compliance trend over time")

line_chart = plot_compliance_rate_over_time(
    df=agg_plot,
    year_col="Data Year",
    group_col=category_col,
    value_col=value_col,
    y_title=y_title,
)
st.altair_chart(line_chart, use_container_width=True)

# --- Plot 2: faceted compliance distributions by year ---
st.subheader("Compliance distribution by category and year")

facet_chart = plot_compliance_status_facets(
    df=agg_plot,
    year_col="Data Year",
    group_col=category_col,
    n_submitted_col="n_submitted",
    n_exempt_col="n_exempt",
    n_not_submitted_col="n_not_submitted",
)
st.altair_chart(facet_chart, use_container_width=True)
