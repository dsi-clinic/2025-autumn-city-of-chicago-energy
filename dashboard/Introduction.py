"""Main file for running dashboard"""

import logging

import streamlit as st

from utils.dashboard_utils import apply_page_config
from utils.data_utils import concurrent_buildings
from utils.settings import DATA_DIR

core_dataframe = concurrent_buildings()
apply_page_config()

st.title("City of Chicago - Energy Dashboard")

with st.container(border=True):
    st.markdown("##### Team")

    # Three-column layout for mentors
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**External Mentor**")
        st.markdown("Candice Stauffer")
    with col2:
        st.markdown("**Internal Mentor**")
        st.markdown("David Jacobson")
    with col3:
        st.markdown("**TA**")
        st.markdown("Carter Tran")

    st.markdown("")

    # Student team
    st.markdown("**Student Team**")
    st.markdown("Kiki Mei, Alejandro Orellana, Mira Shi, Han Zhang")

st.divider()

##################################################

st.subheader("Overview")
st.markdown("*Understanding Chicago's Energy Rating Placard Program*")
st.markdown("")

(
    col1,
    col2,
) = st.columns([6, 4])
with col1:
    st.markdown("""
    This dashboard visualizes energy consumption patterns across Chicago's diverse building stock,
    supporting efforts to improve energy efficiency, reduce costs, and inform policy decisions.

    **Key Features:**
    - Track ENERGY STAR Scores across building types
    - Analyze electricity and gas consumption patterns
    - View trends by neighborhood and year
    """)

    st.markdown("")

    st.markdown("""
    ## Description

    The City of Chicago requires large buildings to display energy rating placards, showing how
    efficiently each building uses energy. This project studies how those public ratings have
    affected building performance over time.

    By analyzing data from 2015–2024, we aim to see whether buildings have become more
    energy-efficient and reduced their greenhouse gas emissions since the placards were introduced in 2019.

    Our findings will help the City understand whether the rating system encourages building owners
    to improve energy efficiency, which can save money, cut emissions, and support Chicago's climate goals.
    """)

    st.markdown("")

    st.markdown("""
    ## Problem Statements

    **Does the placard program work?** The City of Chicago wants to know whether its Energy Rating
    Placard program—which makes building energy efficiency publicly visible—has actually led to
    improvements in energy performance across buildings.

    **What drives improvement?** We need to determine which building characteristics (e.g., size, type,
    energy source mix) are most strongly linked to performance improvements over time.

    **Where to focus efforts?** We want to predict which buildings are most likely to improve, so the
    City can better target outreach or incentives.
    """)

    st.markdown("# Data")
    st.markdown("""
    ### Dataset Overview

    - **Volume:** Covers roughly 10 years of data (2015–2024) for thousands of large buildings across Chicago that report energy use annually under the city’s Energy Benchmarking Ordinance.
    - **Type:** Structured, tabular data combining building characteristics (size, type, construction year, location) with annual performance metrics.
    - **Content:** Includes Energy Star scores, Site Energy Use Intensity (EUI), greenhouse gas emissions, electricity and natural gas consumption, and water use for each property.
    - **Other Details:** Enables tracking the same building over time, supporting pre/post-placard comparison and modeling of improvement trends.
    """)

    st.caption(
        "Filtered to the 2,363 buildings (of 3,852 total) that submitted a report "
        "in every year from 2016 to 2023, enabling consistent year-over-year comparison."
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

st.markdown("")
st.divider()
st.markdown("")

st.markdown("## Chicago Energy Benchmarking Dataset")
st.markdown("#### Buildings with Complete Data (2016-2023)")

# -------------------- Dataset Summary Metrics --------------------
# Calculate key statistics for the dataset
if not core_dataframe.empty:
    total_buildings = core_dataframe["ID"].nunique() if "ID" in core_dataframe.columns else len(core_dataframe)
    years_covered = sorted(core_dataframe["Data Year"].unique()) if "Data Year" in core_dataframe.columns else []
    year_range = f"{min(years_covered)}–{max(years_covered)}" if years_covered else "N/A"
    total_records = len(core_dataframe)
    building_types = core_dataframe["Primary Property Type"].nunique() if "Primary Property Type" in core_dataframe.columns else "N/A"

    st.markdown("**Dataset at a Glance**")
    # Display metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Buildings", f"{total_buildings:,}")
    with metric_col2:
        st.metric("Year Range", year_range)
    with metric_col3:
        st.metric("Total Data Points", f"{total_records:,}")
    with metric_col4:
        st.metric("Building Types", building_types)

    st.markdown("---")

# -------------------- Interactive Data Table --------------------
st.markdown(
    "**💡 Tip:** Click column headers to sort, use search to filter, "
    "or scroll to explore the full dataset."
)

st.dataframe(
    core_dataframe,
    use_container_width=True,
    height=400,
    hide_index=True,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
