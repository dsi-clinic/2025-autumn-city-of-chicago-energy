"""Main file for running dashboard"""

import logging

import streamlit as st

from utils.dashboard_utils import apply_page_config
from utils.data_utils import concurrent_buildings

main_dataframe = concurrent_buildings()
apply_page_config()

st.title("City of Chicago - Energy Dashboard")
st.subheader("Mentors & Team")

with st.container(border=True):
    # Three-column layout for mentors
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div style='text-align: center;'>
                <strong>External Mentor:</strong> Candice Stauffer
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style='text-align: center;'>
                <strong>Internal Mentor:</strong> David Jacobson
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style='text-align: center;'>
                <strong>TA:</strong> Carter Tran
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full-page centered team line
    st.markdown(
        """
        <div style='text-align: center; margin-top: 20px;'>
            <strong>Team:</strong> Kiki Mei, Alejandro Orellana, Mira Shi, Han Zhang
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

##################################################

(
    col1,
    col2,
) = st.columns(2)
with col1:
    st.markdown(
        "This dashboard visualizes energy consumption patterns across Chicago’s diverse building stock,"
        "supporting efforts to improve energy efficiency, reduce costs, and inform policy decisions."
        "It aggregates and presents data on key metrics such as ENERGY STAR Scores, electricity use, and gas consumption, broken down by building type, neighborhood, and year."
    )

    st.markdown("## Description")
    st.markdown(
        "- The City of Chicago requires large buildings to display energy rating placards, showing how efficiently each building uses energy. This project studies how those public ratings have affected building performance over time."
    )
    st.markdown(
        "- By analyzing data from 2015–2024, we aim to see whether buildings have become more energy-efficient and reduced their greenhouse gas emissions since the placards were introduced in 2019."
    )
    st.markdown(
        "- Our findings will help the City understand whether the rating system encourages building owners to improve energy efficiency, which can save money, cut emissions, and support Chicago’s climate goals."
    )

    st.markdown("## Problem Statements")
    st.markdown(
        "- The City of Chicago wants to know whether its Energy Rating Placard program—which makes building energy efficiency publicly visible—has actually led to improvements in energy performance across buildings."
    )
    st.markdown(
        "- To support this, we need to determine which building characteristics (e.g., size, type, energy source mix) are most strongly linked to performance improvements over time."
    )
    st.markdown(
        "- We also want to predict which buildings are most likely to improve, so the City can better target outreach or incentives."
    )

    st.markdown("## Data")
    st.markdown(
        "**Volume:** Covers roughly 10 years of data (2015–2024) for thousands of large buildings across Chicago that report energy use annually under the city’s Energy Benchmarking Ordinance."
    )
    st.markdown(
        "**Type:** Structured, tabular data combining building characteristics (size, type, construction year, location) with annual performance metrics."
    )
    st.markdown(
        "**Content:** Includes Energy Star scores, Site Energy Use Intensity (EUI), greenhouse gas emissions, electricity and natural gas consumption, and water use for each property."
    )
    st.markdown(
        "**Other Details:** The dataset allows tracking the same building over time, enabling pre/post-placard comparison and modeling of improvement trends."
    )

    st.markdown(
        "The final dataframe filters dowe data to only have concurrent buildings for 10 year span"
    )
with col2:
    st.image(
        "../data/image/Chicago_River_Aerial.jpg",
        caption="Chicago River Aerial View",
        use_container_width=True,
    )
######################################################

st.divider()

st.markdown("## Dataframe")

st.dataframe(main_dataframe)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
