"""Streamlit page: Non-Compliance Rate by Community Area

- Left: overall non-compliance rate (exempt excluded from rate denominator)
- Right: non-compliance rate for selected property type (dropdown)

Uses:
- utils.dashboard_utils.apply_page_config, cache_full_data, cache_community_geojson_url
- utils.data_utils.add_compliance_status, build_compliance_base_year,
  build_area_table_overall, build_area_table_by_property
- utils.plot_utils.plot_noncompliance_per_year, plot_noncompliance_by_property
"""

import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_community_geojson_url,
    cache_full_data,
)
from utils.data_utils import (
    add_compliance_status,
    build_area_table_by_property,
    build_area_table_overall,
    build_compliance_base_year,
    clean_property_type,
)
from utils.plot_utils import (
    plot_noncompliance_by_property,
    plot_noncompliance_per_year,
)


def main() -> None:
    """Streamlit page for analyzing building compliance and non-compliance rates."""
    apply_page_config()
    st.title("Non-Compliance Rate by Community Area")

    # Load Data
    full_data = cache_full_data()
    full_data = clean_property_type(full_data)
    chi_geo = cache_community_geojson_url()

    years_list = sorted([int(y) for y in full_data["Data Year"].dropna().unique()])

    # -------------------- Controls --------------------
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        year = st.selectbox("Year", years_list, index=0, key="nc_year")

    with c2:
        scheme = st.selectbox(
            "Color scheme",
            options=["blues", "greens", "oranges", "reds", "purples"],
            index=0,
            key="nc_scheme",
        )

    with c3:
        fix_domain = st.checkbox(
            "Fix color scale (0–1) for rate", value=True, key="nc_fix"
        )

    with c4:
        top_n = st.slider(
            "Top N property types (dropdown)",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key="nc_topn",
        )

    domain = (0.0, 1.0) if fix_domain else None

    # -------------------- Build compliance + tables --------------------
    try:
        # NOTE: you must pass the same energy metric column list you use elsewhere
        energy_cols = [
            "Electricity Use (kBtu)",
            "Natural Gas Use (kBtu)",
            "District Steam Use (kBtu)",
            "District Chilled Water Use (kBtu)",
            "All Other Fuel Use (kBtu)",
            "Total GHG Emissions (Metric Tons CO2e)",
            "GHG Intensity (kg CO2e/sq ft)",
            "Site EUI (kBtu/sq ft)",
            "Source EUI (kBtu/sq ft)",
            "Weather Normalized Site EUI (kBtu/sq ft)",
            "Weather Normalized Source EUI (kBtu/sq ft)",
        ]

        data = add_compliance_status(full_data, energy_cols=energy_cols, inplace=False)
        base = build_compliance_base_year(data, year=int(year))
        area_overall = build_area_table_overall(base)
        area_type, top_ptypes = build_area_table_by_property(
            base, top_n_property_types=int(top_n)
        )

        # -------------------- Build charts --------------------
        left = plot_noncompliance_per_year(
            chi_geo=chi_geo,
            area=area_overall,
            scheme=scheme,
            domain=domain,
            width=650,
            height=550,
            year=int(year),
            color_field="non_compliance_rate",
        )

        right = plot_noncompliance_by_property(
            chi_geo=chi_geo,
            area_type=area_type,
            top_ptypes=top_ptypes,
            scheme=scheme,
            domain=domain,
            width=650,
            height=550,
            year=int(year),
            color_field="non_compliance_rate",
        )

        # -------------------- Layout: two maps side-by-side --------------------
        colL, colR = st.columns(2)

        with colL:
            st.altair_chart(left, use_container_width=True)

        with colR:
            st.altair_chart(right, use_container_width=True)

        # -------------------- Table (overall) --------------------
        st.divider()
        st.subheader("Community-area table (overall)")

        area_tbl_display = area_overall.copy()
        area_tbl_display["area_display"] = (
            area_tbl_display["area_display"].astype(str).str.title()
        )

        with st.expander("Show table", expanded=False):
            st.dataframe(
                area_tbl_display.sort_values(
                    ["non_compliance_rate", "denom"], ascending=[False, False]
                ),
                use_container_width=True,
                height=380,
            )

            csv = area_tbl_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download table as CSV",
                data=csv,
                file_name=f"noncompliance_area_overall_{year}.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(
            "Could not render the non-compliance maps. "
            "Common causes: missing columns (energy/status), empty year slice, or geojson key mismatch."
        )
        st.exception(e)


if __name__ == "__main__":
    main()
