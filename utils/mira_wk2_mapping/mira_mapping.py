"""mira_mapping.py.

Functions for visualizing Chicago Energy Benchmarking data using Altair.
Includes choropleth maps, bar charts, and metric change analyses.
"""

import re
from pathlib import Path

import altair as alt
import pandas as pd


def prepare_geojson(geojson: dict) -> pd.DataFrame:
    """Extract neighborhood names and geometry from GeoJSON."""
    return pd.DataFrame(
        {
            "Neighborhood": [f["properties"]["pri_neigh"] for f in geojson["features"]],
            "Alt_Name": [f["properties"]["sec_neigh"] for f in geojson["features"]],
            "geometry": [f["geometry"] for f in geojson["features"]],
        }
    )


def aggregate_metric(dff: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate a given metric by Community Area and Data Year."""
    dff = dff.dropna(subset=["Community Area", metric]).copy()
    dff["Neighborhood"] = dff["Community Area"].str.strip().str.title()
    dff[metric] = pd.to_numeric(dff[metric], errors="coerce")
    grouped = dff.groupby(["Neighborhood", "Data Year"], as_index=False)[metric].mean()
    return grouped


def plot_choropleth(
    geojson: dict, dff: pd.DataFrame, metric: str, year: int | None = None
) -> alt.Chart:
    """Plot a choropleth map for a given metric across Chicago neighborhoods.

    Uses the default Altair projection and automatic sizing.
    """
    if year:
        data = dff[dff["Data Year"] == year]
        title = f"{metric} by Neighborhood ({year})"
    else:
        data = dff.groupby("Neighborhood", as_index=False)[metric].mean()
        title = f"Average {metric} by Neighborhood (All Years)"

    base = (
        alt.Chart(alt.Data(values=geojson["features"]))
        .mark_geoshape(stroke="white", strokeWidth=0.5, fill="lightgrey")
        .project(type="mercator")
        .properties(width=600, height=400)
    )

    overlay = (
        alt.Chart(alt.Data(values=geojson["features"]))
        .mark_geoshape(stroke="white", strokeWidth=0.5)
        .transform_lookup(
            lookup="properties.pri_neigh",
            from_=alt.LookupData(data, "Neighborhood", [metric]),
        )
        .encode(
            color=alt.Color(
                f"{metric}:Q",
                title=metric,
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title=metric),
            ),
            tooltip=[
                alt.Tooltip("properties.pri_neigh:N", title="Neighborhood"),
                alt.Tooltip(f"{metric}:Q", format=".2f", title=metric),
            ],
        )
        .project(type="mercator")
        .properties(title=title)
    )

    return base + overlay


def save_all_metric_maps(
    dff: pd.DataFrame, geojson: dict, metrics: list[str], output_dir: Path
) -> None:
    """Generate and save choropleth grids for all specified metrics.

    Each metric produces one combined grid with an all-year average map and yearly maps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_cols = 4

    for metric in metrics:
        safe_metric = re.sub(r"[^A-Za-z0-9_]", "_", metric).strip("_")
        agg = aggregate_metric(dff, metric)
        charts = [plot_choropleth(geojson, agg, metric)]

        for year in sorted(dff["Data Year"].dropna().unique()):
            charts.append(plot_choropleth(geojson, agg, metric, int(year)))

        grid = []
        for i in range(0, grid_cols * 3, grid_cols):
            row = charts[i : i + grid_cols]
            while len(row) < grid_cols:
                row.append(
                    alt.Chart().mark_text(text="").properties(width=600, height=400)
                )
            grid.append(alt.hconcat(*row))

        combined = alt.vconcat(*grid).resolve_scale(color="independent")
        combined = combined.properties(
            title=f"{metric} by Neighborhood (All Years + Yearly Breakdown)"
        ).configure_title(fontSize=13)

        file_path = output_dir / f"{safe_metric}_grid.png"
        combined.save(str(file_path), scale_factor=2)


def plot_building_count_map(
    geojson: dict, dff: pd.DataFrame, year: int | None = None
) -> alt.Chart:
    """Plot a choropleth map showing the number of buildings per neighborhood.

    If year is specified, plots that year; otherwise plots the all-year average.
    """
    df_counts = (
        dff.dropna(subset=["Community Area", "Data Year"])
        .groupby(["Community Area", "Data Year"], as_index=False)
        .agg(Building_Count=("ID", "count"))
    )

    if year:
        data = df_counts[df_counts["Data Year"] == year]
        title = f"Number of Reported Buildings by Neighborhood ({year})"
    else:
        data = (
            df_counts.groupby("Community Area", as_index=False)["Building_Count"]
            .mean()
            .rename(columns={"Building_Count": "Building_Count"})
        )
        title = "Average Number of Reported Buildings by Neighborhood (All Years)"

    base = (
        alt.Chart(alt.Data(values=geojson["features"]))
        .mark_geoshape(stroke="white", strokeWidth=0.5, fill="lightgrey")
        .project(type="mercator")
        .properties(width=600, height=400)
    )

    overlay = (
        alt.Chart(alt.Data(values=geojson["features"]))
        .mark_geoshape(stroke="white", strokeWidth=0.5)
        .transform_lookup(
            lookup="properties.pri_neigh",
            from_=alt.LookupData(data, "Community Area", ["Building_Count"]),
        )
        .encode(
            color=alt.Color(
                "Building_Count:Q",
                title="Building Count",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title="Number of Buildings"),
            ),
            tooltip=[
                alt.Tooltip("properties.pri_neigh:N", title="Neighborhood"),
                alt.Tooltip("Building_Count:Q", format=".0f", title="Buildings"),
            ],
        )
        .project(type="mercator")
        .properties(title=title)
    )

    return base + overlay


def plot_building_count_bar(dff: pd.DataFrame) -> alt.Chart:
    """Create a faceted bar chart showing the number of reported buildings.

    Displays results by neighborhood and year.
    """
    building_counts = (
        dff.dropna(subset=["Community Area", "Data Year"])
        .groupby(["Community Area", "Data Year"], as_index=False)
        .agg(Building_Count=("ID", "count"))
    )

    total_counts = (
        building_counts.groupby("Community Area")["Building_Count"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    sort_order = list(total_counts)

    bar_base = (
        alt.Chart(building_counts)
        .mark_bar()
        .encode(
            x=alt.X(
                "Community Area:N",
                sort=sort_order,
                axis=alt.Axis(labels=False, ticks=False, title=None),
            ),
            y=alt.Y("Building_Count:Q", title="Number of Buildings"),
            color=alt.Color("Building_Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("Community Area:N", title="Neighborhood"),
                alt.Tooltip("Data Year:O", title="Year"),
                alt.Tooltip("Building_Count:Q", title="Buildings Reported"),
            ],
        )
        .properties(width=250, height=200)
    )

    bar_chart = (
        bar_base.facet(facet=alt.Facet("Data Year:O", title="Year"), columns=3)
        .properties(title="Number of Reported Buildings by Neighborhood and Year")
        .configure_axis(labelFontSize=6, titleFontSize=9)
        .configure_title(fontSize=13)
    )

    return bar_chart


def plot_metric_change_map(
    geojson: dict, dff: pd.DataFrame, metrics: list[str]
) -> list[alt.Chart]:
    """Plot the change in each metric (latest - earliest year) per neighborhood.

    Only includes buildings that reported across multiple years.
    """
    multi_year_ids = (
        dff.groupby("ID")["Data Year"]
        .nunique()
        .reset_index()
        .query("`Data Year` > 1")["ID"]
    )
    df_stable = dff[dff["ID"].isin(multi_year_ids)].copy()

    for m in metrics:
        df_stable[m] = pd.to_numeric(df_stable[m], errors="coerce")

    change_df = (
        df_stable.groupby(["Community Area", "Data Year"])[metrics]
        .mean()
        .groupby("Community Area")
        .apply(lambda x: x.loc[x.index.max()] - x.loc[x.index.min()])
        .reset_index()
        .rename(columns={m: f"{m} Change" for m in metrics})
    )

    charts = []
    for metric in metrics:
        metric_change_col = f"{metric} Change"
        title = f"Change in {metric} (Latest - Earliest Year, Multi-Year Buildings)"

        base = (
            alt.Chart(alt.Data(values=geojson["features"]))
            .mark_geoshape(stroke="white", strokeWidth=0.5, fill="lightgrey")
            .project(type="mercator")
            .properties(width=600, height=400)
        )

        overlay = (
            alt.Chart(alt.Data(values=geojson["features"]))
            .mark_geoshape(stroke="white", strokeWidth=0.5)
            .transform_lookup(
                lookup="properties.pri_neigh",
                from_=alt.LookupData(change_df, "Community Area", [metric_change_col]),
            )
            .encode(
                color=alt.Color(
                    f"{metric_change_col}:Q",
                    title=f"{metric} Change",
                    scale=alt.Scale(scheme="redblue", domainMid=0),
                    legend=alt.Legend(title=f"{metric} Change"),
                ),
                tooltip=[
                    alt.Tooltip("properties.pri_neigh:N", title="Neighborhood"),
                    alt.Tooltip(f"{metric_change_col}:Q", format=".2f", title="Change"),
                ],
            )
            .project(type="mercator")
            .properties(title=title)
        )

        charts.append(base + overlay)

    return charts
