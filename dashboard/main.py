"""Main file for running dashboard"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import loading_data

main_dataframe = loading_data()

st.set_page_config(page_title="Chicago Energy Dashboard", layout="wide")

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

lines = (
    alt.Chart(sch)
    .mark_line(strokeWidth=2)
    .encode(x="Data Year", y="mean(Electricity Use (kBtu))", color="Community Area:N")
)

left, center, right = st.columns([1, 2, 1])
with center:
    st.altair_chart(lines, use_container_width=True)


st.divider()
