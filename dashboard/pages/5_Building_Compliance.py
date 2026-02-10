"""Building Compliance Over Time Visualization"""

import streamlit as st
import pandas as pd

from utils.data_utils import (
    assign_effective_year_built,
    categorize_time_built,
    clean_property_type,
    concurrent_buildings,
    load_data,
    add_reporting_compliance_flags,
    clean_year_built
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


# --- Load & prepare data ---
energy_df = load_data()
energy_df = clean_year_built(energy_df)
energy_df = assign_effective_year_built(energy_df)
energy_df = clean_property_type(energy_df)
energy_df = categorize_time_built(energy_df)

# Apply standardized compliance logic (2018+)
energy_df = add_reporting_compliance_flags(energy_df)

# Restrict to 2018+ (already done inside helper, but safe to be explicit)
energy_df = energy_df[energy_df["Data Year"] >= 2018]


# Optional: restrict to years where ordinance is active and data is consistent
min_year, max_year = 2014, 2023
energy_df = energy_df[
    (energy_df["Data Year"] >= min_year) & (energy_df["Data Year"] <= max_year)
]


# --- Compliance flag ---
# Assumes your merge filled Reporting Status = "Not present in data"
# for covered-but-unreported buildings
energy_df["Is_Compliant"] = energy_df["Reporting Status"].ne("Not present in data")


# --- Filter controls ---
variables = [
    "Is_Compliant",
    "Chicago Energy Rating",
    "ENERGY STAR Score",
]

category_options = ["Time Built", "Primary Property Type", "Community Area"]
category_col = st.selectbox(
    "Select category for Building Classification",
    options=category_options,
    index=category_options.index("Time Built"),
)

metric_option = st.selectbox(
    "Compliance metric",
    options=[
        "Share of buildings compliant",
        "Number of compliant buildings",
        "Number of non‑compliant buildings",
    ],
    index=0,
)

col1, col2, col3 = st.columns(3)

with col1:
    time_built_opts = sorted(energy_df["Time Built"].dropna().unique().tolist())
    sel_time_built = st.multiselect(
        "Time Built", time_built_opts, default=time_built_opts
    )

with col2:
    ppt_opts = sorted(energy_df["Primary Property Type"].dropna().unique().tolist())
    sel_ppt = st.multiselect("Primary Property Type", ppt_opts, default=ppt_opts)

with col3:
    ca_opts = sorted(energy_df["Community Area"].dropna().unique().tolist())
    sel_ca = st.multiselect("Community Area", ca_opts, default=ca_opts)


energy_df_filtered = energy_df[
    energy_df["Time Built"].isin(sel_time_built)
    & energy_df["Primary Property Type"].isin(sel_ppt)
    & energy_df["Community Area"].isin(sel_ca)
]


if energy_df_filtered.empty:
    st.warning(
        "No buildings match the selected filters. Please broaden your selections."
    )
    st.stop()


if energy_df_filtered["Data Year"].nunique() < 2:
    st.warning(
        "Not enough years of data for the selected filters to evaluate compliance trends."
    )
    st.stop()


# --- Aggregate compliance over time ---
# year × category × status
group_cols = ["Data Year", category_col]

agg = (
    energy_df
    .groupby(group_cols, dropna=False)
    .agg(
        n_buildings=("ID", "nunique"),
        n_submitted=("SubmittedFlag", "sum"),
        n_exempt=("ExemptFlag", "sum"),
        n_not_submitted=("NotSubmittedFlag", "sum"),
        n_non_compliant=("NonCompliantFlag", "sum"),
    )
    .reset_index()
)

agg["share_submitted"] = agg["n_submitted"] / agg["n_buildings"].where(agg["n_buildings"] > 0)
agg["share_non_compliant"] = agg["n_non_compliant"] / agg["n_buildings"].where(agg["n_buildings"] > 0)



# --- Choose metric to plot ---
metric_option = st.selectbox(
    "Compliance metric",
    options=[
        "Share submitted",
        "Share non‑compliant",
        "Number submitted",
        "Number non‑compliant",
    ],
    index=0,
)

if metric_option == "Share submitted":
    value_col = "share_submitted"
    y_title = "Share submitted"
elif metric_option == "Share non‑compliant":
    value_col = "share_non_compliant"
    y_title = "Share non‑compliant"
elif metric_option == "Number submitted":
    value_col = "n_submitted"
    y_title = "Submitted buildings"
else:
    value_col = "n_non_compliant"
    y_title = "Non‑compliant buildings"



class_opts = sorted(agg[category_col].dropna().unique().tolist())
selected_class = st.selectbox(f"{category_col} filter", ["All"] + class_opts)

if selected_class != "All":
    agg_plot = agg[agg[category_col] == selected_class]
else:
    agg_plot = agg


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
