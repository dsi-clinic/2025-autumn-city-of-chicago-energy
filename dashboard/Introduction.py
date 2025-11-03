"""Main file for running dashboard"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.dashboard_utils import apply_page_config
from utils.data_utils import load_data

main_dataframe = load_data()

apply_page_config()

st.title("City of Chicago - Energy Dashboard")
st.subheader("Mentors & Team")

with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **External Mentor:** Candice Stauffer
        """)
    with col2:
        st.markdown("""
        **Internal Mentor:** David Jacobson
        """)
    with col3:
        st.markdown("""
        **TA:** Carter Tran
        """)

    center = st.columns([1, 2, 1])[1]
    with center:
        st.markdown("**Team:** Kiki Mei, Alejandro Orellana, Mira Shi, Han Zhang")
st.divider()

Chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])

st.subheader("Area Chart")
st.area_chart(Chart_data)

st.subheader("Line Chart")
st.line_chart(Chart_data)

st.header("Map")

map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [41.8781, -87.6298], columns=["lat", "lon"]
)

st.map(map_data)

st.divider()

st.dataframe(main_dataframe)

sch = main_dataframe[main_dataframe["Primary Property Type"] == "college/university"]

alt.data_transformers.disable_max_rows()

hover = alt.selection_point(
    fields=["Community Area"], on="mouseover", clear=True, nearest=True
)

lines = (
    alt.Chart(sch)
    .mark_line(strokeWidth=2)
    .encode(
        x="Data Year:O",
        y="mean(Electricity Use (kBtu)):Q",
        color="Community Area:N",
        tooltip=["Community Area", "Data Year", "mean(Electricity Use (kBtu))"],
    )
)

points = (
    alt.Chart(sch)
    .mark_point(size=50)
    .encode(
        x="Data Year:O",
        y="mean(Electricity Use (kBtu)):Q",
        color="Community Area:N",
        tooltip=["Community Area", "Data Year", "mean(Electricity Use (kBtu))"],
    )
)

lines = (
    (lines + points)
    .add_params(hover)
    .encode(opacity=alt.condition(hover, alt.value(1), alt.value(0.2)))
    .interactive()
)

left, center, right = st.columns([1, 2, 1])
with center:
    st.altair_chart(lines, use_container_width=True)


st.divider()
