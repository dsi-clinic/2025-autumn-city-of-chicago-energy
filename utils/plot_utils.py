"""plot_utils.py

This module provides plotting utilities for visualizing energy benchmarking data.
It includes reusable functions for comparing variable distributions before and after 2019,
 (with support for log-scaled y-axes and clean, publication-ready visuals),
 Year-grouped histograms / density plots for numeric variables,
 Building-level energy delta and cumulative change metrics,
 Choropleth maps using Altair and metric change analyses.
"""

import logging
import re
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

# ----------Comparable analysis (Bar charts and delta charts)-----------


def compare_variable_distribution(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    variable: str,
    label1: str = "Before 2019",
    label2: str = "After 2019",
    log_scale: bool = False,
) -> None:
    """Visualize the distribution of a numeric variable across two DataFrames using boxplots + stripplots.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataset (e.g., energy_df_pre_2020)
    df2 : pd.DataFrame
        Second dataset (e.g., energy_df_post_2019)
    variable : str
        Name of the numeric column to compare
    label1 : str
        Label for the first dataset
    label2 : str
        Label for the second dataset
    log_scale : bool, optional
        Whether to use a logarithmic y-scale. Defaults to False.

    Returns:
    -------
    None
        This function generates and displays a plot but does not return a value.
    """
    if variable not in df1.columns or variable not in df2.columns:
        raise ValueError(f"Variable '{variable}' not found in both DataFrames.")

    df1_temp = df1[[variable]].copy()
    df1_temp["Period"] = label1

    df2_temp = df2[[variable]].copy()
    df2_temp["Period"] = label2

    combined_df = pd.concat([df1_temp, df2_temp], ignore_index=True)
    combined_df = combined_df.dropna(subset=[variable])

    # Plot
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(8, 6))

    ax = sns.boxplot(
        data=combined_df, x="Period", y=variable, showfliers=False, width=0.5
    )
    sns.stripplot(
        data=combined_df, x="Period", y=variable, color="black", size=2, alpha=0.3
    )

    # Log scale option
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Axis & layout improvements
    plt.title(f"{variable} Comparison: {label1} vs {label2}", fontsize=14, pad=15)
    plt.xlabel("")
    plt.ylabel(variable)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    palette: str | None = None,
    figsize: tuple = (8, 5),
    legend_title: str | None = None,
    rotate_xticks: int | None = 45,
    show_values: bool = False,
) -> None:
    """Create a customizable bar plot with consistent styling.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing data to plot.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    hue : str, optional
        Column name for color grouping (default is None).
    title : str, optional
        Title of the plot (default is None).
    xlabel : str, optional
        Label for the x-axis (default uses the column name).
    ylabel : str, optional
        Label for the y-axis (default uses the column name).
    palette : str, optional
        Seaborn color palette name (default is "Blues_d").
    figsize : tuple, optional
        Figure size in inches (default is (8, 5)).
    legend_title : str, optional
        Custom title for the legend (default uses the hue column name).
    rotate_xticks : int, optional
        Degrees to rotate x-axis labels (default is 45).
    show_values : bool, optional
        Whether to display numeric labels on top of each bar (default is False).

    Returns:
    -------
    None
        Displays the bar plot.
    """
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=data, x=x, y=y, hue=hue, palette=palette)

    # Add title and labels
    ax.set_title(title or "", fontsize=14)
    ax.set_xlabel(xlabel or x, fontsize=12)
    ax.set_ylabel(ylabel or y, fontsize=12)

    # Rotate x-axis labels if needed
    if rotate_xticks:
        plt.xticks(rotation=rotate_xticks)

    # Add gridlines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add legend if applicable
    if hue:
        plt.legend(
            title=legend_title or hue,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False,
        )

    # Optionally show values on bars
    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", fontsize=9, padding=3)

    plt.tight_layout()
    plt.show()


def plot_building_energy_deltas(
    pivot_df: pd.DataFrame,
    metric_name: str = "Energy Use",
    id_col: str = "ID",
    start_year: int = None,
    end_year: int = None,
    marker_year: int = 2019,
    figsize: tuple[float, float] = (18, 6),
    alpha_buildings: float = 0.3,
    linewidth_buildings: float = 1,
) -> None:
    """Generate side-by-side plots for building-level energy delta trends for any metric.

    Includes:
      1. Average Year-over-Year Δ
      2. Year-over-Year Δ per building with mean trend
      3. Cumulative Δ from baseline with mean trend

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Pivoted DataFrame with buildings as rows and years as columns.
    metric_name : str, default="Energy Use"
        Descriptive name of the metric to display on titles and axis labels.
    id_col : str, default="ID"
        Column name for building identifier used when melting.
    marker_year : int, default=2019
        Year to highlight with a vertical line (e.g., placards introduction).
    """
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    subset_years = [year for year in pivot_df.columns if start_year <= year <= end_year]
    pivot_df = pivot_df[subset_years]
    num_years = len(subset_years)

    # --- Compute year-over-year Δ
    delta_df = pivot_df.diff(axis=1)

    # --- Melt for building-level plotting
    melted_delta = delta_df.reset_index().melt(
        id_vars=id_col, var_name="Data Year", value_name=f"Δ {metric_name}"
    )

    # --- Compute cumulative change from baseline
    baseline_year = start_year
    cumulative_change = pivot_df.subtract(pivot_df[start_year], axis=0)
    melted_cum = cumulative_change.reset_index().melt(
        id_vars=id_col, var_name="Data Year", value_name=f"Δ {metric_name}"
    )

    # --- Mean trends
    mean_delta = delta_df.mean(axis=0)
    mean_cum = cumulative_change.mean(axis=0)

    # --- Set up 3 side-by-side subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ---- 1️⃣ Average Year-over-Year Δ ----
    axes[0].plot(mean_delta.index, mean_delta.values, marker="o", linewidth=2)
    axes[0].axvline(
        x=marker_year,
        color="red",
        linestyle="--",
        label=f"Placards introduced {marker_year}",
    )
    axes[0].set_title(f"Average Year-over-Year Δ {metric_name}")
    axes[0].set_xlabel("Data Year")
    axes[0].set_ylabel(f"Δ {metric_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # ---- 2️⃣ Building-level Year-over-Year Δ ----
    sns.lineplot(
        data=melted_delta,
        x="Data Year",
        y=f"Δ {metric_name}",
        hue=id_col,
        alpha=alpha_buildings,
        linewidth=linewidth_buildings,
        legend=False,
        ax=axes[1],
    )
    axes[1].plot(
        mean_delta.index,
        mean_delta.values,
        color="black",
        linewidth=2,
        label="Mean Trend",
    )
    axes[1].axvline(
        x=marker_year,
        color="red",
        linestyle="--",
        label=f"Placards introduced {marker_year}",
    )
    axes[1].set_title(f"Year-over-Year Δ {metric_name} per Building")
    axes[1].set_xlabel("Data Year")
    axes[1].set_ylabel(f"Δ {metric_name}")
    axes[1].legend()

    # ---- 3️⃣ Cumulative Δ from baseline ----
    sns.lineplot(
        data=melted_cum,
        x="Data Year",
        y=f"Δ {metric_name}",
        hue=id_col,
        alpha=alpha_buildings,
        linewidth=linewidth_buildings,
        legend=False,
        ax=axes[2],
    )
    axes[2].plot(
        mean_cum.index, mean_cum.values, color="black", linewidth=2, label="Mean Trend"
    )
    axes[2].axvline(
        x=marker_year,
        color="red",
        linestyle="--",
        label=f"Placards introduced {marker_year}",
    )
    axes[2].set_title(f"Cumulative Δ {metric_name} per Building (from {baseline_year})")
    axes[2].set_xlabel("Data Year")
    axes[2].set_ylabel(f"Δ {metric_name}")
    axes[2].legend()

    # --- Big title and layout
    fig.suptitle(
        f"Buildings with {num_years} Years of Data ({start_year}-{end_year})\nMetric: {metric_name}",
        fontsize=16,
        y=1.05,
    )

    plt.tight_layout()
    plt.show()


# ----------Line graphs-----------


def plot_trend_by_year(
    df: pd.DataFrame, numeric_cols: list[str], agg: str = "median"
) -> None:
    """Plot yearly trends (mean or median) for numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'Data Year' or 'Data_Year'.
    numeric_cols : list
        Columns to plot trends for.
    agg : str
        Aggregation function to use ('mean' or 'median').
    """
    year_col = "Data Year" if "Data Year" in df.columns else "Data_Year"

    for col in numeric_cols:
        if col not in df.columns:
            continue

        plt.figure(figsize=(8, 5))
        if agg == "median":
            trend = df.groupby(year_col)[col].median()
        else:
            trend = df.groupby(year_col)[col].mean()

        trend.plot(marker="o", linestyle="-", color="steelblue")
        plt.title(f"{agg.capitalize()} {col} Over Time")
        plt.xlabel("Year")
        plt.ylabel(col)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


def plot_mean_cumulative_changes(
    metrics_dict: dict[str, pd.DataFrame],
    start_year: int | None = None,
    end_year: int | None = None,
    marker_year: int = 2019,
    title_prefix: str = "Cumulative % Change from Baseline",
) -> None:
    """Plot average cumulative % change from baseline for multiple energy metrics.

    metrics_dict : dict
        { 'Metric Name': DataFrame_of_percent_changes }
    """
    plt.figure(figsize=(10, 6))

    for label, metric_df in metrics_dict.items():
        metric_df = metric_df.copy()
        metric_df.columns = metric_df.columns.astype(int)
        metric_df = metric_df.reindex(sorted(metric_df.columns), axis=1)
        mean_changes = metric_df.mean(axis=0)
        cum = (1 + mean_changes / 100).cumprod() - 1
        cum *= 100
        plt.plot(cum.index, cum.values, marker="o", linewidth=2, label=label)

    plt.axvline(
        x=marker_year,
        color="red",
        linestyle="--",
        label=f"Placards introduced ({marker_year})",
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(
        f"{title_prefix} ({start_year}-{end_year})", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Cumulative % Change from Baseline", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ----------Histograms (data exploration)-----------


def plot_facet_histograms_by_year(
    df: pd.DataFrame, variable: str, bins: int = 40
) -> None:
    """Faceted histograms with shared x/y axes to maintain consistent scale."""
    year_col = "Data Year" if "Data Year" in df.columns else "Data_Year"
    if variable not in df.columns:
        logging.warning(f"'{variable}' column not found.")
        return

    x_min, x_max = df[variable].min(), df[variable].max()

    g = sns.FacetGrid(
        df,
        col=year_col,
        col_wrap=4,
        sharex=True,
        sharey=True,
        height=3.5,
    )
    g.map_dataframe(sns.histplot, x=variable, bins=bins, color="skyblue")
    g.set_titles(col_template="Year: {col_name}")
    g.set_axis_labels(variable, "Count")
    g.set(xlim=(x_min, x_max))
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(
        f"Distribution of {variable} by Year (Consistent Scale)", fontsize=14
    )
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    plot_facet_histograms_by_year()


# ----------Spatial Mapping-----------


def prepare_geojson(geojson: dict) -> pd.DataFrame:
    """Extract neighborhood names and geometry from GeoJSON."""
    return pd.DataFrame(
        {
            "Neighborhood": [f["properties"]["pri_neigh"] for f in geojson["features"]],
            "Alt_Name": [f["properties"]["sec_neigh"] for f in geojson["features"]],
            "geometry": [f["geometry"] for f in geojson["features"]],
        }
    )


# help plottting spatial maps by aggregate mean metrics
def aggregate_metric(dff: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate a given metric by Community Area and Data Year using Mean."""
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
