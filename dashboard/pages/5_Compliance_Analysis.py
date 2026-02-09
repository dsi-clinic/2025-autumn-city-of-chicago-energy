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
    plot_noncompliance_by_property_selected,
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

    # -------------------- Controls --------------------
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])

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
            "Fix color scale (0–1) for rate",
            value=True,
            key="nc_fix",
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

    # -------------------- Compute compliance + tables --------------------
    try:
        # Add/normalize compliance_status once (vectorized, uses your rule)
        data = add_compliance_status(
            energy_data=full_data,
            energy_cols=energy_cols,
            inplace=False,
        )

        # Build year base (one row per building-area-ptype for the year)
        base = build_compliance_base_year(
            energy_data=data,
            year=int(year),
        )

        # Overall area table (includes exempt counts, but denom excludes exempt)
        area_overall = build_area_table_overall(base)

        # Area × property type table (exempt excluded in this one)
        area_type, top_ptypes = build_area_table_by_property(
            base,
            top_n_property_types=int(top_n),
        )

        #  Property Type Selector (Full Width)
        if top_ptypes:
            selected_ptype = st.selectbox(
                "Select property type",
                options=top_ptypes,
                index=0,
                key="nc_ptype",
            )
        else:
            selected_ptype = None
            st.selectbox(
                "Select property type",
                options=["(none available)"],
                disabled=True,
            )

        # -------------------- Build charts --------------------
        overall_chart = plot_noncompliance_per_year(
            chi_geo=chi_geo,
            area=area_overall,
            scheme=scheme,
            domain=domain,
            width=650,
            height=550,
            year=int(year),
            color_field="non_compliance_rate",
        )

        # Right chart depends on whether we have property types
        if selected_ptype is None:
            property_chart = None
        else:
            property_chart = plot_noncompliance_by_property_selected(
                chi_geo=chi_geo,
                area_type=area_type,
                selected_ptype=selected_ptype,
                scheme=scheme,
                domain=domain,
                width=650,
                height=550,
                year=int(year),
                color_field="non_compliance_rate",
            )

        # -------------------- Layout: side-by-side maps --------------------
        left, right = st.columns(2)

        with left:
            st.altair_chart(overall_chart, use_container_width=True)

        with right:
            if property_chart is None:
                st.info("No property types available for this year after cleaning.")
            else:
                st.altair_chart(property_chart, use_container_width=True)

        # -------------------- Table (overall) --------------------
        st.markdown("### Community-area table (overall)")

        with st.expander("Show table", expanded=False):
            # nice sorting: highest rate first, then denom
            tbl = area_overall.copy()
            if "non_compliance_rate" in tbl.columns:
                tbl = tbl.sort_values(
                    ["non_compliance_rate", "denom"],
                    ascending=[False, False],
                )
            st.dataframe(tbl, use_container_width=True, height=360)

            csv = tbl.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download table as CSV",
                data=csv,
                file_name=f"noncompliance_area_overall_{year}.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(
            "Could not render the non-compliance maps. Common causes: missing columns "
            "(energy/status), empty year slice, or geojson key mismatch."
        )
        st.exception(e)


if __name__ == "__main__":
    main()
