"""Main file for running dashboard"""

import logging

import streamlit as st

from utils.dashboard_utils import apply_page_config
from utils.data_utils import concurrent_buildings
from utils.settings import DATA_DIR

core_dataframe = concurrent_buildings()
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

    st.markdown("""
    ## Description

    - The City of Chicago requires large buildings to display energy rating placards, showing how efficiently each building uses energy. This project studies how those public ratings have affected building performance over time.
    - By analyzing data from 2015–2024, we aim to see whether buildings have become more energy-efficient and reduced their greenhouse gas emissions since the placards were introduced in 2019.
    - Our findings will help the City understand whether the rating system encourages building owners to improve energy efficiency, which can save money, cut emissions, and support Chicago’s climate goals.
    """)

    st.markdown("""
    ## Problem Statements

    - The City of Chicago wants to know whether its Energy Rating Placard program—which makes building energy efficiency publicly visible—has actually led to improvements in energy performance across buildings.
    - To support this, we need to determine which building characteristics (e.g., size, type, energy source mix) are most strongly linked to performance improvements over time.
    - We also want to predict which buildings are most likely to improve, so the City can better target outreach or incentives.
    """)

    st.markdown("# Data")
    st.markdown("""
    ### Dataset Overview

    - **Volume:** Covers roughly 10 years of data (2015–2024) for thousands of large buildings across Chicago that report energy use annually under the city’s Energy Benchmarking Ordinance.
    - **Type:** Structured, tabular data combining building characteristics (size, type, construction year, location) with annual performance metrics.
    - **Content:** Includes Energy Star scores, Site Energy Use Intensity (EUI), greenhouse gas emissions, electricity and natural gas consumption, and water use for each property.
    - **Other Details:** Enables tracking the same building over time, supporting pre/post-placard comparison and modeling of improvement trends.
    """)

    st.markdown(
        "**The final dataframe filters down data to only have core 2363 out of 3852 buildings for the 10 year span**"
    )

    st.markdown("""
    #### Definitions

    - **Chicago Energy Rating:** The zero-to-four-star Chicago Energy Rating assigned to the building in the shown Data Year. A building with zero stars did not submit a report, or did submit a report but was missing required information. All other buildings receive between one and four stars, with four stars reflecting the highest performance. Every building receives a Chicago Energy Rating Placard with this rating, which must be posted in a prominent location at the building. The rating must also be shared at the time of listing the building for sale or for lease. For more information, visit: www.ChicagoEnergyRating.org. This column was added for the 2018 Data Year. It is blank for previous years.
    - **ENERGY STAR Score:** 1–100 rating that assesses a property’s overall energy performance, based on national data to control for differences among climate, building uses, and operations. A score of 50 represents the national median.
    - **Exempt From Chicago Energy Rating:** Shows whether the building is subject to the Chicago Energy Rating Ordinance. Some properties are required to submit energy benchmarking reports but are not subject to the requirements of the Chicago Energy Rating program. These buildings do not receive a Chicago Energy Rating, typically due to technical reasons. This column was added for the 2018 Data Year. It is blank for previous years.
    """)

with col2:
    st.image(
        DATA_DIR / "image" / "Chicago_River_Aerial.jpg",
        caption="Chicago River Aerial View",
        use_container_width=True,
    )

with st.expander("Timeline of Reported Data"):
    st.markdown("""
    - [**2014** Data Reported in 2015](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2014-Data-Reported-in-/tepd-j7h5/about_data)
    - [**2015** Data Reported in 2016](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2015-Data-Reported-in-/ebtp-548e/about_data)
    - [**2016** Data Reported in 2017](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2016-Data-Reported-in-/fpwt-snya/about_data)
    - [**2017** Data Reported in 2018](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2017-Data-Reported-in-/j2ev-2azp/about_data)
    - [**2018** Data Reported in 2019 *(First year with Chicago Energy Rating)*](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2018-Data-Reported-in-/m2kv-bmi3/about_data)
    - [**2019** Data Reported in 2020](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2019-Data-Reported-in-/jn94-it7m/about_data)
    - [**2020** Data Reported in 2021](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2020-Data-Reported-in-/ydbk-8hi6/about_data)
    - [**2021** Data Reported in 2023](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2021-Data-Reported-in-/gkf4-txtp/about_data)
    - [**2022** Data Reported in 2023](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2022-Data-Reported-in-/mz3g-jagv/about_data)
    - [**2023** Data Reported in 2024](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2023-Data-Reported-in-/3a36-5x9a/about_data)
    """)

######################################################

st.divider()

st.markdown("## Core Dataframe")

st.dataframe(core_dataframe)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
