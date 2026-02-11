"""Streamlit page: Non-Compliance Rate by Community Area (with Property Type dropdown)

Drop this file into:
  dashboard/pages/Non_Compliance_Map.py   (or similar)

It uses:
- utils.dashboard_utils.apply_page_config, cache_full_data, year_lists, cache_community_geojson
- utils.plot_utils.noncompliance_choropleth_by_year
"""

import pandas as pd
import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_community_geojson_url,
    cache_full_data,
)
from utils.plot_utils import noncompliance_choropleth_by_year


def build_energy_reported(full_df: pd.DataFrame) -> pd.DataFrame:
    """Define 'reported' rows for compliance rule (IDs counted as compliant).

    NOTE: load_data() lowercases string cols; Reporting Status becomes lower-case strings.
    Some missing values become literal "nan" strings because of astype(str).
    """
    if "Reporting Status" not in full_df.columns:
        return full_df.iloc[0:0].copy()

    s = full_df["Reporting Status"].astype(str).str.strip().str.lower()
    allowed = {"submitted", "submitted data", "nan"}
    return full_df[s.isin(allowed)].copy()


def main() -> None:
    """Display the choropleth of non-compliance rate by Community area and property types"""
    apply_page_config()
    st.title("Non-Compliance Rate by Community Area")

    # -------------------- Load Data --------------------
    full_data = cache_full_data()
    # chi_geo = cache_community_geojson()
    chi_geo = cache_community_geojson_url()

    years_list = sorted(
        [int(year) for year in sorted(full_data["Data Year"].dropna().unique())]
    )

    # Reported subset used as "compliant IDs" (your compliance rule)
    energy_reported = build_energy_reported(full_data)

    # -------------------- Controls --------------------
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])

    with ctrl1:
        year = st.selectbox("Year", years_list, index=0, key="nc_year")

    with ctrl2:
        color_field = st.selectbox(
            "Map metric",
            options=["non_compliance_rate", "non_compliant", "compliant", "total"],
            index=0,
            key="nc_metric",
        )

    with ctrl3:
        top_n = st.slider(
            "Top N property types (dropdown)",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key="nc_topn",
        )

    with ctrl4:
        fix_domain = st.checkbox(
            "Fix color scale (0–1) for rate", value=True, key="nc_fix"
        )
        scheme = st.selectbox(
            "Color scheme",
            options=["blues", "greens", "oranges", "reds", "purples"],
            index=0,
            key="nc_scheme",
        )

    domain = (
        (0.0, 1.0) if (fix_domain and color_field == "non_compliance_rate") else None
    )

    # -------------------- Chart --------------------
    try:
        chart, area_type_table = noncompliance_choropleth_by_year(
            energy_data=full_data,  # denominator: all IDs present in that year
            energy_reported=energy_reported,  # compliant IDs: reported
            chi_geo=chi_geo,  # COMMUNITY AREA geojson
            year=int(year),
            area_name_col="Community Area",
            geo_area_name_key="properties.community",
            property_type_col="Primary Property Type",
            top_n_property_types=int(top_n),
            color_field=color_field,
            scheme=scheme,
            domain=domain,
            width=650,
            height=550,
        )

        st.altair_chart(chart, use_container_width=True)

        area_tbl_display = area_type_table.copy()
        area_tbl_display["Community Area"] = area_tbl_display[
            "Community Area"
        ].str.title()
        area_tbl_display["Property Type"] = area_tbl_display[
            "Property Type"
        ].str.title()

        with st.expander("Show community-area table", expanded=False):
            st.dataframe(
                area_tbl_display.sort_values(
                    ["Property Type", "non_compliance_rate", "total"],
                    ascending=[True, False, False],
                ),
                use_container_width=True,
                height=360,
            )

            csv = area_tbl_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download table as CSV",
                data=csv,
                file_name=f"noncompliance_area_propertytype_{year}.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(
            "Could not render the non-compliance map. "
            "Most common causes: geojson key mismatch, empty data after cleaning, or missing columns."
        )
        st.exception(e)


if __name__ == "__main__":
    main()
