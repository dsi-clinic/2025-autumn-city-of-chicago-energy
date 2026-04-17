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
    ## Questions This Dashboard Explores

    **1. Is the placard program working?**
    Do buildings improve their energy efficiency after ratings become public?

    **2. What drives improvement?**
    Which building characteristics (size, type, age) predict better performance?

    **3. Where should the city focus?**
    Which buildings are most likely to benefit from outreach or incentives?
    """)

    st.markdown("")

    with st.container(border=True):
        st.markdown("""
        > **What is Energy Benchmarking?**
        > Chicago requires large buildings (50,000+ sq ft) to report their energy use annually.
        > This data helps track energy consumption, identify inefficient buildings, and measure
        > progress toward climate goals.
        """)

    st.markdown("# Data")
    st.markdown("""
    ### About the Data

    This dashboard analyzes 10 years (2015-2024) of energy reports from thousands of large Chicago buildings.

    **Includes:**
    - Energy consumption (electricity, gas, steam)
    - ENERGY STAR scores
    - Greenhouse gas emissions
    - Building characteristics (size, type, age, location)
    """)

    st.markdown(
        "**Filtered to:** 2,363 buildings that reported consistently from 2016-2023, "
        "enabling year-over-year comparisons."
    )

    st.markdown("""
    #### Definitions

    **Chicago Energy Rating:** A 0-4 star rating displayed on building placards.

    - **4 stars** = Highest efficiency
    - **1 star** = Lowest efficiency
    - **0 stars** = Building did not report data or had incomplete data

    **Required since 2019:** Buildings must post this rating in lobbies and share it when selling or leasing.
    For more information, visit: www.ChicagoEnergyRating.org.

    **ENERGY STAR Score:** 1–100 rating that assesses a property's overall energy performance, based on
    national data to control for differences among climate, building uses, and operations. A score of 50
    represents the national median.

    **Exempt from Rating:** Some buildings must still report energy data but don't receive a star rating,
    typically due to technical reasons (e.g., unusual energy systems, special use buildings).
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
