"""Minimal GeoJSON render test

Goal:
- Confirm GeoJSON loads
- Confirm Altair can render polygons
- Nothing else
"""

import altair as alt
import streamlit as st

from utils.dashboard_utils import (
    apply_page_config,
    cache_community_geojson,
    cache_community_geojson_url,
)


def main() -> None:
    """Render a minimal Streamlit page to test GeoJSON map rendering."""
    apply_page_config()
    st.title("GeoJSON Minimal Render Test")

    # Load geojson using data URL
    geo_url = cache_community_geojson_url()
    data = alt.Data(
        url=geo_url, format=alt.DataFormat(type="json", property="features")
    )

    # Load geojson
    geo = cache_community_geojson()
    feats = geo.get("features", [])
    st.write(f"Loaded **{len(feats)}** features.")

    # ---- Inspect keys ----
    with st.expander("Inspect properties keys (first feature)", expanded=True):
        if feats:
            props0 = feats[0].get("properties", {}) or {}
            st.write("Property keys found:")
            st.code(", ".join(sorted(props0.keys())))
            st.write("First feature properties (raw):")
            st.json(props0)
        else:
            st.warning("GeoJSON has no features.")

    # PURE geoshape — no encode, no transform, no color
    chart = (
        alt.Chart(data)  # use data here instead of inline unwrap
        .mark_geoshape(
            filled=False,
            stroke="#444",
            strokeWidth=1.0,
        )
        .project(type="mercator")
        .properties(
            width=900,
            height=650,
            title="Community Areas (outlines only)",
        )
    )

    st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
