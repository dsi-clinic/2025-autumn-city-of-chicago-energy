"""Compliance Over Time + Non-Compliance Maps (Shared Filters)

Updated behavior:
- Charts use the 4 small filters (Time Built / PPT / TLPT / Community Area).
- Maps IGNORE the 4 small filters and only depend on:
  (1) Map year, and
  (2) The selected classification category (category_col) + selected group (right-map selector).
- If category_col == "Community Area", we show a notice and only render the LEFT map
  (because the choropleth is already grouped by community area).
"""

from __future__ import annotations

import streamlit as st

from utils.dashboard_utils import (
    aggregate_compliance_over_time,
    apply_category_filter,
    apply_page_config,
    build_standard_filters,
    cache_community_geojson_url,
    cache_full_data_prepped,
    choose_compliance_metric,
    filter_energy_by_selections,
    load_clean_energy_data_for_dashboards,
    show_helpful_filter_error,
)
from utils.data_utils import (
    add_reporting_compliance_flags,
    add_top_level_property_type,
    build_area_table_by_property,
    build_area_table_overall,
    build_compliance_base,
)
from utils.plot_utils import (
    plot_compliance_rate_over_time,
    plot_compliance_status_facets,
    plot_noncompliance_by_property_selected,
    plot_noncompliance_per_year,
)


# -----------------------------
# Helpers
# -----------------------------
def _pretty_metric_label(value_col: str, y_title: str) -> str:
    return y_title if y_title else value_col.replace("_", " ").title()


def _metric_to_map_field(value_col: str) -> tuple[str, tuple[float, float] | None]:
    """Map chart metric (choose_compliance_metric output) -> map table column.

    Map tables from build_area_table_overall / build_area_table_by_property include:
      - share_submitted, share_non_compliant
      - num_submitted, num_non_compliant
      - denom
    """
    if value_col in {"share_submitted", "share_non_compliant"}:
        return value_col, (0.0, 1.0)

    if value_col in {"num_submitted", "num_non_compliant", "denom"}:
        return value_col, None

    aliases = {
        "n_submitted": "num_submitted",
        "n_non_compliant": "num_non_compliant",
        "n_not_submitted": "num_non_compliant",
        "n_total": "denom",
    }
    if value_col in aliases:
        return aliases[value_col], None

    return "share_non_compliant", (0.0, 1.0)


# -----------------------------
# Page
# -----------------------------
def main() -> None:
    """Render the Compliance Combine dashboard page.

    This page:
    - Shows compliance trends over time (benchmarking-only dataset).
    - Provides community-area non-compliance maps (merged dataset).
    - Shares the same classification category and compliance metric selection.
    - Charts use the four small filters; maps ignore them and only use year + category.
    """
    apply_page_config()
    st.title("Compliance with Chicago Energy Benchmarking Over Time")

    st.markdown(
        """
This dashboard tracks Chicago buildings’ energy benchmarking compliance over time,
and adds community-area **non-compliance maps** under the **same metric** selection.
"""
    )

    # -------------------- Load datasets --------------------
    restriction_year = 2018

    # (A) Charts dataset: benchmarking-only cleaned data
    energy_df = load_clean_energy_data_for_dashboards(
        min_year=2014,
        max_year=2023,
        restrict_to_concurrent=False,
    )
    energy_df = add_top_level_property_type(energy_df)
    energy_df = add_reporting_compliance_flags(energy_df)
    energy_df = energy_df[energy_df["Data Year"] >= restriction_year].copy()

    # (B) Maps dataset: covered × benchmarking merged data
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
    full_data, years_list = cache_full_data_prepped(energy_cols=energy_cols)
    chi_geo = cache_community_geojson_url()

    # Restrict years list + data for maps to >= restriction_year
    full_data = full_data[full_data["Data Year"] >= restriction_year].copy()
    years_list = sorted(
        [int(y) for y in full_data["Data Year"].dropna().unique().tolist()]
    )

    # -------------------- (1) Category selector (used by both maps and charts) --------------------
    st.subheader("Select Classification Category")
    category_options = [
        "Primary Property Type",
        "Top-Level Property Type",
        "Time Built",
        "Community Area",
    ]
    category_col = st.selectbox(
        "Classify buildings by:",
        category_options,
        index=0,
        key="combine_category_col",
    )

    # ============================================================
    # (2)-(5) Maps section - DISPLAYED FIRST (ignores 4 small filters below)
    # ============================================================
    st.divider()
    st.subheader("Non-compliance maps (community area)")

    # -------------------- (2) Map year selector --------------------
    DEFAULT_YEAR = 2018
    default_year_index = (
        years_list.index(DEFAULT_YEAR)
        if DEFAULT_YEAR in years_list
        else len(years_list) - 1
    )

    map_year = st.selectbox(
        "Year",
        years_list,
        index=max(0, default_year_index),
        key="combine_map_year",
    )

    # Maps ignore the 4 small filters — ONLY filter by year
    df_year = full_data.loc[full_data["Data Year"] == int(map_year)].copy()
    if df_year.empty:
        st.warning(
            "No buildings available for the chosen map year in the merged dataset."
        )
        st.stop()

    # -------------------- (3) Compliance metric (shared with charts) --------------------
    value_col, y_title = choose_compliance_metric()
    metric_label = _pretty_metric_label(value_col, y_title)

    # -------------------- Build base + overall table --------------------
    base = build_compliance_base(df_year, year=int(map_year))
    area_overall = build_area_table_overall(base)

    # -------------------- Metric for maps --------------------
    map_color_field, map_domain = _metric_to_map_field(value_col)
    if map_color_field not in area_overall.columns:
        map_color_field, map_domain = "share_non_compliant", (0.0, 1.0)
        metric_label = "Share non-compliant"

    # -------------------- Community Area special-case --------------------
    if category_col == "Community Area":
        st.info(
            "ℹ️ **Community Area selected**: the choropleth is already grouped by community area, "
            "so a second grouped map cannot be generated. Showing the overall map only."
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Overall (all buildings)")
            st.altair_chart(
                plot_noncompliance_per_year(
                    chi_geo=chi_geo,
                    area=area_overall,
                    scheme="blues",
                    domain=map_domain,
                    width=650,
                    height=550,
                    year=int(map_year),
                    color_field=map_color_field,
                    title=f"{metric_label} — Overall ({map_year})",
                ),
                use_container_width=True,
            )

        with right:
            st.empty()

        with st.expander("Show community-area table (overall)", expanded=False):
            st.dataframe(area_overall, use_container_width=True, height=350)

    else:
        # -------------------- (4) Group selector for maps --------------------
        area_type = build_area_table_by_property(
            base,
            ptype_col=category_col,
            status_col="compliance_status",
        )

        ptype_options = sorted(
            area_type["ptype_key"].dropna().astype("string").unique().tolist()
        )
        if not ptype_options:
            st.warning("No groups available for the selected category in this year.")
            st.stop()

        selected_map_group = st.selectbox(
            f"{category_col} (for right map)",
            ptype_options,
            index=0,
            key="combine_map_group",
        )

        # -------------------- (5) Two maps --------------------
        left, right = st.columns(2)

        with left:
            st.subheader("Overall (all buildings)")
            st.altair_chart(
                plot_noncompliance_per_year(
                    chi_geo=chi_geo,
                    area=area_overall,
                    scheme="blues",
                    domain=map_domain,
                    width=650,
                    height=550,
                    year=int(map_year),
                    color_field=map_color_field,
                    title=f"{metric_label} — Overall ({map_year})",
                ),
                use_container_width=True,
            )

        with right:
            st.subheader(f"{category_col}: {selected_map_group}")
            st.altair_chart(
                plot_noncompliance_by_property_selected(
                    chi_geo=chi_geo,
                    area_type=area_type,
                    selected_ptype=selected_map_group,
                    scheme="blues",
                    domain=map_domain,
                    width=650,
                    height=550,
                    year=int(map_year),
                    color_field=map_color_field,
                    title=f"{metric_label} — {selected_map_group} ({map_year})",
                ),
                use_container_width=True,
            )

        with st.expander("Show community-area table (overall)", expanded=False):
            st.dataframe(area_overall, use_container_width=True, height=350)

    # ============================================================
    # (6)-(8) Charts section - DISPLAYED SECOND (uses 4 small filters)
    # ============================================================
    st.divider()
    st.subheader("Compliance trends over time (filtered)")

    # -------------------- (6) 4 small filters (CHARTS ONLY) --------------------
    sel_time_built, sel_ppt, sel_tlpt, sel_ca = [None, None, None, None]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        time_built_options = ["All"] + sorted(
            energy_df["Time Built"].dropna().unique().tolist()
        )
        sel_time_built = st.selectbox(
            "Time Built",
            time_built_options,
            index=0,
            key="combine_time_built",
        )

    with col2:
        ppt_options = ["All"] + sorted(
            energy_df["Primary Property Type"].dropna().unique().tolist()
        )
        sel_ppt = st.selectbox(
            "Primary Property Type",
            ppt_options,
            index=0,
            key="combine_ppt",
        )

    with col3:
        tlpt_options = ["All"] + sorted(
            energy_df["Top-Level Property Type"].dropna().unique().tolist()
        )
        sel_tlpt = st.selectbox(
            "Top-Level Property Type",
            tlpt_options,
            index=0,
            key="combine_tlpt",
        )

    with col4:
        ca_options = ["All"] + sorted(
            energy_df["Community Area"].dropna().unique().tolist()
        )
        sel_ca = st.selectbox(
            "Community Area",
            ca_options,
            index=0,
            key="combine_ca",
        )

    # -------------------- (7) Apply the standard filter logic to the charts dataset --------------------
    energy_df_filtered = filter_energy_by_selections(
        energy_df,
        sel_time_built=sel_time_built,
        sel_ppt=sel_ppt,
        sel_ca=sel_ca,
        sel_tlpt=sel_tlpt,
    )

    if energy_df_filtered.empty:
        # Show helpful error message with actionable suggestions
        filter_selections = {}
        if sel_time_built != "All":
            filter_selections["Time Built"] = sel_time_built
        if sel_ppt != "All":
            filter_selections["Primary Property Type"] = sel_ppt
        if sel_tlpt != "All":
            filter_selections["Top-Level Property Type"] = sel_tlpt
        if sel_ca != "All":
            filter_selections["Community Area"] = sel_ca

        show_helpful_filter_error(energy_df_filtered, energy_df, filter_selections)
        st.stop()

    MIN_YEARS_REQUIRED = 2
    if energy_df_filtered["Data Year"].nunique() < MIN_YEARS_REQUIRED:
        st.warning("Not enough years of data under these filters to evaluate trends.")
        st.stop()

    # -------------------- (8) Category-value filter (All vs one class) --------------------
    agg = aggregate_compliance_over_time(
        energy_df=energy_df_filtered,
        category_col=category_col,
    )
    agg_plot, selected_class = apply_category_filter(agg, category_col)

    # -------------------- (9) Charts --------------------
    st.altair_chart(
        plot_compliance_rate_over_time(
            df=agg_plot,
            year_col="Data Year",
            group_col=category_col,
            value_col=value_col,
            y_title=y_title,
        ),
        use_container_width=True,
    )

    st.altair_chart(
        plot_compliance_status_facets(
            df=agg_plot,
            year_col="Data Year",
            group_col=category_col,
            n_submitted_col="n_submitted",
            n_exempt_col="n_exempt",
            n_not_submitted_col="n_not_submitted",
        ),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
