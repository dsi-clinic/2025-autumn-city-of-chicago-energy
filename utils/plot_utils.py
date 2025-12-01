"""plot_utils.py

This module provides plotting utilities for visualizing energy benchmarking data.
It includes reusable functions for comparing variable distributions before and after 2019,
 (with support for log-scaled y-axes and clean, publication-ready visuals),
 Year-grouped histograms / density plots for numeric variables,
 Building-level energy delta and cumulative change metrics,
 Choropleth maps using Altair and metric change analyses.
"""

import logging
import math
import re
import warnings
from collections.abc import Callable, Iterable

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
) -> tuple:
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
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes objects for further customization or saving.

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
    fig, ax = plt.subplots(figsize=(8, 6))

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

    return fig, ax


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
) -> tuple:
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
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes objects for further customization or saving.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax)

    ax.set_title(title or "", fontsize=14)
    ax.set_xlabel(xlabel or x, fontsize=12)
    ax.set_ylabel(ylabel or y, fontsize=12)

    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_xticks)

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if hue:
        ax.legend(
            title=legend_title or hue,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False,
        )

    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", fontsize=9, padding=3)

    fig.tight_layout()
    return fig, ax


def plot_delta_divergence(
    df: pd.DataFrame,
    ptype: str,
    year_before: int,
    year_after: int,
    eui_col: str = "Weather Normalized Site EUI (kBtu/sq ft)",
    property_col: str = "Primary Property Type",
    name_col: str = "Property Name",
) -> alt.Chart:
    """Interactive Altair bar chart showing ΔEUI between two years for a given property type.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing building-level annual EUI values.
    ptype : str
        Target property type to filter.
    year_before : int
        The baseline year for comparison.
    year_after : int
        The comparison year for measuring change.
    eui_col : str, default "Weather Normalized Site EUI (kBtu/sq ft)"
        Column name containing EUI values.
    property_col : str, default "Primary Property Type"
        Column identifying property type for filtering.
    name_col : str, default "Property Name"
        Column for building names to display in tooltips.

    Returns:
    -------
    alt.Chart
        An interactive Altair bar chart visualizing ΔEUI sorted by magnitude,
        with hover tooltips showing building ID, name, EUI before/after, and ΔEUI.
    """
    df_type = df[df[property_col].str.lower() == ptype.lower()].copy()
    df_subset = df_type[df_type["Data Year"].isin([year_before, year_after])]
    df_subset = df_subset[["ID", name_col, "Data Year", eui_col]].copy()

    pivot = df_subset.pivot_table(
        index=["ID", name_col], columns="Data Year", values=eui_col
    ).dropna(subset=[year_before, year_after])

    pivot.columns = pivot.columns.astype(str)
    pivot = pivot.reset_index()
    before_str = str(year_before)
    after_str = str(year_after)
    pivot["delta"] = pivot[after_str] - pivot[before_str]

    pivot = pivot.sort_values("delta").reset_index(drop=True)
    pivot["xpos"] = pivot.index
    pivot["color"] = pivot["delta"].apply(lambda x: "red" if x > 0 else "blue")

    chart = (
        alt.Chart(pivot)
        .mark_bar()
        .encode(
            x=alt.X("xpos:O", title="Buildings (sorted by ΔEUI)"),
            y=alt.Y("delta:Q", title=f"Δ {eui_col}"),
            color=alt.Color("color:N", scale=None),
            tooltip=[
                alt.Tooltip("ID:N", title="Building ID"),
                alt.Tooltip(name_col + ":N", title="Property Name"),
                alt.Tooltip(before_str + ":Q", title=f"EUI {year_before}"),
                alt.Tooltip(after_str + ":Q", title=f"EUI {year_after}"),
                alt.Tooltip("delta:Q", title="ΔEUI"),
            ],
        )
        .properties(
            width=850,
            height=350,
            title=f"Divergence of ΔEUI for {ptype.title()} Buildings ({year_after} – {year_before})",
        )
        .interactive()
    )

    return chart


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
) -> tuple:
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


    Returns:
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes objects for further customization or saving.
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
    return fig, axes


# ----------Line graphs-----------


def plot_multi_line_trend(
    df: pd.DataFrame,
    year_col: str,
    value_cols: list[str],
    labels: list[str] | None = None,
    title: str = "Trend Over Time",
    ylabel: str = "",
    marker_year: int | None = None,
    figsize: tuple = (8, 5),
) -> tuple:
    """Plot multiple lines (e.g., Chicago vs National) on the same chart.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing year and metric columns.
    year_col : str
        Name of the year column.
    value_cols : list[str]
        Columns to plot as separate lines.
    labels : list[str], optional
        Pretty labels for legend. If None, use value_cols.
    title : str
        Chart title.
    ylabel : str
        Label for y-axis.
    marker_year : int, optional
        Draw a vertical dashed line (e.g., 2019 for placards).
    figsize : tuple
        Figure size.

    Returns:
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axis objects.
    """
    if labels is None:
        labels = value_cols

    fig, ax = plt.subplots(figsize=figsize)

    for col, label in zip(value_cols, labels):
        ax.plot(
            df[year_col],
            df[col],
            marker="o",
            linewidth=2,
            label=label,
        )

    if marker_year is not None:
        ax.axvline(
            marker_year,
            color="gray",
            linestyle="--",
            alpha=0.6,
            label=f"Policy year {marker_year}",
        )

    # ---- Styling ----
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_relative_change(
    data: pd.DataFrame,
    year_col: str,
    nat_col: str,
    chi_col: str,
    title: str,
    ylabel: str = "Percent Change (%)",
    baseline_year: int = 2018,
) -> tuple:
    """Plot relative percent change (vs baseline) for National vs Chicago."""
    data = data.copy()

    # Baseline values
    base_nat = data.loc[data[year_col] == baseline_year, nat_col].to_numpy()[0]
    base_chi = data.loc[data[year_col] == baseline_year, chi_col].to_numpy()[0]

    # Compute %
    data["Nat_Percent_Change"] = (data[nat_col] / base_nat - 1) * 100
    data["Chi_Percent_Change"] = (data[chi_col] / base_chi - 1) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        data[year_col],
        data["Nat_Percent_Change"],
        marker="o",
        label="National % change",
    )
    ax.plot(
        data[year_col],
        data["Chi_Percent_Change"],
        marker="s",
        label="Chicago % change",
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(2019, color="gray", linestyle="--", alpha=0.6, label="Placard starts")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_trend_by_year(
    df: pd.DataFrame, numeric_cols: list[str], agg: str = "median"
) -> list[tuple]:
    """Plot yearly trends (mean or median) for numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'Data Year' or 'Data_Year'.
    numeric_cols : list
        Columns to plot trends for.
    agg : str
        Aggregation function to use ('mean' or 'median').

    Returns:
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes objects for further customization or saving.

    """
    figs = []
    year_col = "Data Year" if "Data Year" in df.columns else "Data_Year"

    for col in numeric_cols:
        if col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        trend = (
            df.groupby(year_col)[col].median()
            if agg == "median"
            else df.groupby(year_col)[col].mean()
        )
        ax.plot(trend.index, trend.values, marker="o", linestyle="-", color="steelblue")
        ax.set_title(f"{agg.capitalize()} {col} Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        figs.append((fig, ax))

    return figs


def plot_mean_cumulative_changes(
    metrics_dict: dict[str, pd.DataFrame],
    start_year: int | None = None,
    end_year: int | None = None,
    marker_year: int = 2019,
    title_prefix: str = "Cumulative % Change from Baseline",
) -> tuple:
    """Plot average cumulative % change from baseline for multiple energy metrics.

    metrics_dict : dict
        { 'Metric Name': DataFrame_of_percent_changes }

    Returns:
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes objects for further customization or saving.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

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
    return fig, ax


def plot_metric_by_property(
    df: pd.DataFrame,
    metric_col: str,
    agg_func: Callable = pd.Series.median,
    property_col: str = "Primary Property Type",
    year_col: str = "Data Year",
    marker_year: int = 2019,
    width: int = 600,
    height: int = 400,
) -> alt.Chart:
    """Plot an aggregated energy metric (mean, median, etc.) over time by type, with an optional policy marker (e.g. 2019 placard introduction).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing energy metrics, property type, and year columns.
    metric_col : str
        Column name for the energy metric to visualize (e.g., 'Site EUI (kBtu/sq ft)').
    agg_func : Callable, optional
        Aggregation function (e.g., pd.Series.mean, pd.Series.median). Defaults to median.
    property_col : str, optional
        Column name for type. Defaults to 'Primary Property Type'.
    year_col : str, optional
        Column name for data year. Defaults to 'Data Year'.
    marker_year : int, optional
        Year to highlight with a vertical marker (default: 2019).
    width : int, optional
        Chart width (default: 600).
    height : int, optional
        Chart height (default: 400).

    Returns:
    -------
    alt.Chart
        Interactive Altair line chart showing the aggregated metric trends by property type.
    """
    # --- Aggregate data by year and property type ---
    grouped = (
        df.groupby([year_col, property_col], as_index=False)
        .agg({metric_col: agg_func})
        .rename(columns={metric_col: "Aggregated_Metric"})
    )

    # Determine the name of the aggregation function for chart title
    agg_name = getattr(agg_func, "__name__", str(agg_func))

    # --- Build main line chart ---
    line = (
        alt.Chart(grouped)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{year_col}:O", title="Year", sort="ascending"),
            y=alt.Y("Aggregated_Metric:Q", title=f"{agg_name.title()} {metric_col}"),
            color=alt.Color(f"{property_col}:N", title=property_col),
            tooltip=[
                alt.Tooltip(property_col, title=property_col),
                alt.Tooltip(year_col, title="Year"),
                alt.Tooltip(
                    "Aggregated_Metric:Q",
                    format=".2f",
                    title=f"{agg_name.title()} {metric_col}",
                ),
            ],
        )
    )

    # --- Marker line for policy year ---
    marker = (
        alt.Chart(pd.DataFrame({year_col: [marker_year]}))
        .mark_rule(color="red", strokeDash=[4, 4], size=2)
        .encode(x=f"{year_col}:O")
    )

    # --- Annotation label ---
    annotation = (
        alt.Chart(
            pd.DataFrame(
                {
                    year_col: [marker_year],
                    "label": ["Chicago Energy Placard Introduced"],
                }
            )
        )
        .mark_text(align="left", baseline="bottom", dx=5, dy=-5, color="red")
        .encode(x=f"{year_col}:O", text="label:N")
    )

    # --- Combine all layers ---
    chart = (
        (line + marker + annotation)
        .properties(
            width=width,
            height=height,
            title=f"{agg_name.title()} {metric_col} by {property_col} ({df[year_col].min()}–{df[year_col].max()})",
        )
        .interactive()
    )

    return chart


def plot_delta_property_chart(
    df: pd.DataFrame,
    metric_col: str = "Site EUI (kBtu/sq ft)",
    property_col: str = "Primary Property Type",
    id_col: str = "ID",
    year_col: str = "Data Year",
    top_types: Iterable[str] | None = None,
    marker_year: int = 2019,
    width: int = 700,
    height: int = 400,
) -> alt.Chart:
    """Clean data, compute year-over-year change per building, and visualize Δ (year-to-year change) by property type with median trend and 2019 marker.

    Parameters
    ----------
    df : pd.DataFrame
        Raw energy benchmarking dataframe.
    metric_col : str, default="Site EUI (kBtu/sq ft)"
        Energy metric column to compute deltas from.
    property_col : str, default="Primary Property Type"
        Column for property type.
    id_col : str, default="ID"
        Unique building identifier column.
    year_col : str, default="Data Year"
        Column representing the reporting year.
    top_types : iterable of str, optional
        List of property types to show in the dropdown (if None, inferred from df).
    marker_year : int, default=2019
        Year to mark with a red dashed vertical line (Chicago Energy Placard introduction).
    width : int, default=700
        Chart width.
    height : int, default=400
        Chart height.

    Returns:
    -------
    alt.Chart
        Interactive Altair layered chart.
    """
    # Data preperation
    cols = [id_col, year_col, property_col, metric_col]
    df_clean = df[cols].dropna().copy()

    df_clean[year_col] = df_clean[year_col].astype(int)
    df_clean[id_col] = df_clean[id_col].astype(str)
    df_clean[property_col] = df_clean[property_col].astype(str)
    df_clean[metric_col] = pd.to_numeric(df_clean[metric_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[metric_col])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df_delta = (
            df_clean.sort_values([id_col, year_col])
            .groupby(id_col, group_keys=False)
            .apply(
                lambda g: g.assign(Delta=g[metric_col].diff()),
                include_groups=True,
            )
            .dropna(subset=["Delta"])
            .reset_index(drop=True)
        )

    if top_types is None:
        top_types = sorted(df_clean[property_col].unique())

    property_select = alt.selection_point(
        fields=[property_col],
        bind=alt.binding_select(options=list(top_types), name="Property Type: "),
        empty="all",
    )

    # Base line chart (per-building)
    delta_chart = (
        alt.Chart(df_delta)
        .mark_line(opacity=0.25)
        .encode(
            x=alt.X(f"{year_col}:O", title="Year", sort="ascending"),
            y=alt.Y("Delta:Q", title=f"Δ {metric_col}"),
            color=alt.Color(f"{property_col}:N", title="Property Type"),
            detail=f"{id_col}:N",
            tooltip=[
                id_col,
                property_col,
                year_col,
                alt.Tooltip("Delta:Q", format=".2f"),
            ],
        )
        .transform_filter(property_select)
        .add_params(property_select)
        .properties(width=width, height=height)
    )

    # Marker for key policy year
    marker_line = (
        alt.Chart(pd.DataFrame({"x": [marker_year]}))
        .mark_rule(color="red", strokeDash=[4, 4], size=2)
        .encode(x="x:O")
    )

    # Median line for selected property type
    median_line = (
        alt.Chart(df_delta)
        .transform_filter(property_select)
        .transform_aggregate(
            median_delta="median(Delta)",
            groupby=[property_col, year_col],
        )
        .mark_line(size=4, opacity=1.0)
        .encode(
            x=alt.X(f"{year_col}:O", title="Year", sort="ascending"),
            y=alt.Y("median_delta:Q", title=f"Median Δ {metric_col}"),
            color=alt.Color(
                f"{property_col}:N",
                scale=alt.Scale(scheme="tableau10", domain=list(top_types)),
                title="Property Type",
            ),
            tooltip=[
                property_col,
                year_col,
                alt.Tooltip(
                    "median_delta:Q", format=".2f", title=f"Median Δ {metric_col}"
                ),
            ],
        )
    )

    final_chart = (
        (delta_chart + median_line + marker_line)
        .properties(
            title=f"Year-over-Year Change in {metric_col} by Building (Δ from Previous Year)"
        )
        .interactive()
    )

    return final_chart


def plot_did_trend(
    df: pd.DataFrame,
    year_col: str,
    group_col: str,
    outcome_col: str,
    policy_year: int,
    group_labels: dict[int, str],
    title: str = "Difference-in-Differences Trend",
) -> alt.Chart:
    """Plot pre- and post-policy trends for treatment and control groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing year, group, and outcome columns.
    year_col : str
        Column name for the time/year variable.
    group_col : str
        Binary group indicator (0 = control, 1 = treated).
    outcome_col : str
        Column name of the outcome variable.
    policy_year : int
        Year when the policy/intervention took effect.
    group_labels : dict[int, str]
        Mapping of group indicator values (0,1) to readable labels.
    title : str
        Chart title.

    Returns:
    -------
    alt.Chart
        Altair line chart showing DiD group trends over time.
    """
    trend_df = (
        df.groupby([year_col, group_col], as_index=False)[outcome_col]
        .mean()
        .rename(columns={outcome_col: "MeanOutcome"})
    )
    trend_df[group_col] = trend_df[group_col].map(group_labels)

    line = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{year_col}:O", title="Year"),
            y=alt.Y("MeanOutcome:Q", title=outcome_col),
            color=alt.Color(f"{group_col}:N", title="Group"),
            tooltip=[
                alt.Tooltip(f"{year_col}:O", title="Year"),
                alt.Tooltip(f"{group_col}:N", title="Group"),
                alt.Tooltip("MeanOutcome:Q", title="Mean Outcome", format=".2f"),
            ],
        )
    )

    vline = (
        alt.Chart(pd.DataFrame({"x": [policy_year]}))
        .mark_rule(strokeDash=[6, 4], color="red")
        .encode(x="x:O")
    )

    chart = (line + vline).properties(
        title=title,
        width=500,
        height=300,
    )

    return chart


def plot_delta_kernel_density(
    df: pd.DataFrame,
    property_type: str,
    metric_col: str = "Weather Normalized Site EUI (kBtu/sq ft)",
    clip_range: int = 200,
    width: int = 650,
    height: int = 350,
) -> alt.Chart:
    """Interactive KDE distribution plot of Δ metric (year-over-year difference), comparing Pre-2019 vs Post-2019 periods for a selected property type.

    Parameters
    ----------
    df : pd.DataFrame
        Full energy dataset.
    property_type : str
        Building type to filter (e.g., "office", "multifamily housing").
    metric_col : str
        Column name of the metric to compute Δ from.
    clip_range : int
        Range to clip extreme values for readability.
    width : int
        Chart width.
    height : int
        Chart height.

    Returns:
    -------
    alt.Chart
        Interactive KDE plot.
    """
    data = df[df["Primary Property Type"].str.lower() == property_type.lower()].copy()

    if data.empty:
        raise ValueError(f"No buildings found for property type '{property_type}'")

    policy_year = 2019
    data = data.sort_values(["ID", "Data Year"])
    data["Delta"] = data.groupby("ID")[metric_col].diff()
    data = data.dropna(subset=["Delta"])
    data["Period"] = data["Data Year"].apply(
        lambda x: "Pre-2019" if x < policy_year else "Post-2019"
    )

    data["Delta_clipped"] = data["Delta"].clip(-clip_range, clip_range)
    zoom = alt.selection_interval(bind="scales")

    base = alt.Chart(data).transform_density(
        density="Delta_clipped",
        groupby=["Period"],
        as_=["Delta", "Density"],
        extent=[-clip_range, clip_range],
        steps=300,
    )

    chart = (
        base.mark_line()
        .encode(
            x=alt.X("Delta:Q", title=f"Δ {metric_col}"),
            y=alt.Y("Density:Q", title="Density"),
            color=alt.Color("Period:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                alt.Tooltip("Period:N", title="Period"),
                alt.Tooltip("Delta:Q", title="Δ Metric", format=".2f"),
                alt.Tooltip("Density:Q", title="Density", format=".4f"),
            ],
        )
        .add_params(zoom)
        .properties(
            title=f"Shift in Δ Distribution for {property_type.title()} Buildings (Pre vs Post 2019)",
            width=width,
            height=height,
        )
    )

    return chart


# ----------Regression Line plots----------


def plot_energy_persistence_by_year(
    df_lagged: pd.DataFrame,
    property_col: str = "Primary Property Type",
    id_col: str = "ID",
    year_col: str = "Data Year",
    delta_col: str = "Delta",
    delta_next_col: str = "Delta_next",
    start_year: int = 2017,
    end_year: int = 2023,
    width: int = 320,
    height: int = 320,
) -> alt.Chart:
    """Create a 2×3 grid of scatter plots showing Δ N→N+1 vs Δ N+1→N+2 per base year N.

    Parameters
    ----------
    df_lagged : pd.DataFrame
        DataFrame containing lagged Δ values, where each row corresponds to a building-year
        and includes Δ (N→N+1) and Δ_next (N+1→N+2).
    property_col : str, default "Primary Property Type"
        Column name indicating the property type used for coloring and filtering.
    id_col : str, default "ID"
        Column identifying the building (shown in tooltip).
    year_col : str, default "Data Year"
        Column indicating the reference year used to compute lagged differences.
        The base year N is inferred as (year_col - 1).
    delta_col : str, default "Delta"
        Column for ΔEUI from year N→N+1.
    delta_next_col : str, default "Delta_next"
        Column for ΔEUI from year N+1→N+2.
    start_year : int, default 2017
        Earliest base year N to include in the grid.
    end_year : int, default 2023
        Latest base year N to include in the grid.
    width : int, default 320
        Width of each subplot.
    height : int, default 320
        Height of each subplot.

    Returns:
    -------
    alt.Chart
        A vertically concatenated grid (2×3) of interactive Altair scatter plots with:
        - Dots representing buildings
        - Property-type color encoding
        - Per-type regression trend lines
        - Dropdown selector for property type
        - Shared axes across plots
    """
    data = df_lagged.dropna(subset=[delta_col, delta_next_col]).copy()
    data["N_year"] = data[year_col].astype(int) - 1

    data = data[(data["N_year"] >= start_year) & (data["N_year"] <= end_year)]

    years = sorted(data["N_year"].unique().tolist())

    type_select = alt.selection_point(
        fields=[property_col],
        bind=alt.binding_select(
            options=sorted(data[property_col].unique().tolist()),
            name="Property Type: ",
        ),
        empty="all",
    )

    def make_chart(year: int) -> alt.Chart:
        df_year = data[data["N_year"] == year]
        if df_year.empty:
            return alt.Chart().mark_text(text="").properties(width=width, height=height)
        scatter = (
            alt.Chart(df_year)
            .mark_circle(size=55)
            .encode(
                x=alt.X(f"{delta_col}:Q", title="Δ Year N→N+1 (kBtu/sq ft)"),
                y=alt.Y(f"{delta_next_col}:Q", title="Δ Year N+1→N+2 (kBtu/sq ft)"),
                color=alt.condition(
                    type_select, f"{property_col}:N", alt.value("lightgray")
                ),
                opacity=alt.condition(type_select, alt.value(0.8), alt.value(0.15)),
                tooltip=[
                    id_col,
                    property_col,
                    year_col,
                    alt.Tooltip(f"{delta_col}:Q", format=".2f"),
                    alt.Tooltip(f"{delta_next_col}:Q", format=".2f"),
                ],
            )
            .add_params(type_select)
            .properties(width=width, height=height, title=str(year))
        )
        reg = (
            alt.Chart(df_year)
            .transform_regression(delta_col, delta_next_col, groupby=[property_col])
            .mark_line(size=2)
            .encode(
                x=f"{delta_col}:Q",
                y=f"{delta_next_col}:Q",
                color=f"{property_col}:N",
                opacity=alt.condition(type_select, alt.value(1), alt.value(0.2)),
            )
        )
        return scatter + reg

    # compute the grid dynamically based on the number of years and readability
    n_cols = 3
    n_years = len(years)
    n_rows = math.ceil(n_years / n_cols)

    grid_years = []
    for r in range(n_rows):
        row_years = years[r * n_cols : (r + 1) * n_cols]
        while len(row_years) < n_cols:
            row_years.append(None)
        grid_years.append(row_years)

    rows = []
    for row_years in grid_years:
        charts = [
            make_chart(y)
            if y is not None
            else alt.Chart().mark_text(text="").properties(width=width, height=height)
            for y in row_years
        ]
        rows.append(alt.hconcat(*charts))

    final_chart = (
        alt.vconcat(*rows)
        .resolve_scale(x="shared", y="shared")
        .properties(
            title=alt.TitleParams(
                text="Energy-Change Persistence by Base Year N (2017–2023)",
                subtitle=[
                    "Each dot = building’s Δ N→N+1 vs Δ N+1→N+2; lines = trend within property type"
                ],
            )
        )
    )

    return final_chart.interactive()

def plot_energy_persistence_by_year(
    df_lagged: pd.DataFrame,
    property_col: str = "Primary Property Type",
    id_col: str = "ID",
    year_col: str = "Data Year",
    delta_col: str = "Delta",
    delta_next_col: str = "Delta_next",
    start_year: int = 2017,
    end_year: int = 2023,
    width: int = 320,
    height: int = 320,
) -> alt.Chart:
    """Create a 2×3 grid of scatter plots showing Δ N→N+1 vs Δ N+1→N+2 per base year N.

    Each dot = building; color = property type; regression lines per property type per year.
    If fewer than 6 years exist, blank placeholders fill remaining cells for layout balance.
    """
    data = df_lagged.dropna(subset=[delta_col, delta_next_col]).copy()
    data["N_year"] = data[year_col].astype(int) - 1

    data = data[(data["N_year"] >= start_year) & (data["N_year"] <= end_year)]

    years = sorted(data["N_year"].unique().tolist())

    type_select = alt.selection_point(
        fields=[property_col],
        bind=alt.binding_select(
            options=sorted(data[property_col].unique().tolist()),
            name="Property Type: ",
        ),
        empty="all",
    )

    def make_chart(year: int) -> alt.Chart:
        df_year = data[data["N_year"] == year]
        if df_year.empty:
            return alt.Chart().mark_text(text="").properties(width=width, height=height)
        scatter = (
            alt.Chart(df_year)
            .mark_circle(size=55)
            .encode(
                x=alt.X(f"{delta_col}:Q", title="Δ Year N→N+1 (kBtu/sq ft)"),
                y=alt.Y(f"{delta_next_col}:Q", title="Δ Year N+1→N+2 (kBtu/sq ft)"),
                color=alt.condition(
                    type_select, f"{property_col}:N", alt.value("lightgray")
                ),
                opacity=alt.condition(type_select, alt.value(0.8), alt.value(0.15)),
                tooltip=[
                    id_col,
                    property_col,
                    year_col,
                    alt.Tooltip(f"{delta_col}:Q", format=".2f"),
                    alt.Tooltip(f"{delta_next_col}:Q", format=".2f"),
                ],
            )
            .add_params(type_select)
            .properties(width=width, height=height, title=str(year))
        )
        reg = (
            alt.Chart(df_year)
            .transform_regression(delta_col, delta_next_col, groupby=[property_col])
            .mark_line(size=2)
            .encode(
                x=f"{delta_col}:Q",
                y=f"{delta_next_col}:Q",
                color=f"{property_col}:N",
                opacity=alt.condition(type_select, alt.value(1), alt.value(0.2)),
            )
        )
        return scatter + reg

    grid_years = [
        years[0:3],
        years[3:5],
    ]

    row_n = 3

    for row in grid_years:
        while len(row) < row_n:
            row.append(None)

    rows = []
    for row_years in grid_years:
        charts = [
            make_chart(y)
            if y is not None
            else alt.Chart().mark_text(text="").properties(width=width, height=height)
            for y in row_years
        ]
        rows.append(alt.hconcat(*charts))

    final_chart = (
        alt.vconcat(*rows)
        .resolve_scale(x="shared", y="shared")
        .properties(
            title=alt.TitleParams(
                text="Energy-Change Persistence by Base Year N (2017–2023)",
                subtitle=[
                    "Each dot = building’s Δ N→N+1 vs Δ N+1→N+2; lines = trend within property type"
                ],
            )
        )
    )

    return final_chart.interactive()

def plot_energy_persistence_rows(
    df_lagged: pd.DataFrame,
    property_col: str = "Primary Property Type",
    id_col: str = "ID",
    year_col: str = "Data Year",
    delta_col: str = "Delta",
    delta_next_col: str = "Delta_next",
    start_year: int = 2017,
    end_year: int = 2023,
    width: int = 320,
    height: int = 320,
    selected_category: str = None,
) -> list[alt.HConcatChart]:
    """Return a list of row charts (each row is an hconcat of years)."""

    data = df_lagged.dropna(subset=[delta_col, delta_next_col]).copy()
    if selected_category is not None:
        data = data[data[property_col] == selected_category]
    data["N_year"] = data[year_col].astype(int) - 1
    data = data[(data["N_year"] >= start_year) & (data["N_year"] <= end_year)]

    years = sorted(data["N_year"].unique().tolist())
    if not years:
        return []

    type_select = alt.selection_point(
        fields=[property_col],
        empty="all",
    )

    def make_chart(year: int) -> alt.Chart:
        df_year = data[data["N_year"] == year]
        if df_year.empty:
            return alt.Chart().mark_text(text="").properties(width=width, height=height)

        scatter = (
            alt.Chart(df_year)
            .mark_circle(size=55)
            .encode(
                x=alt.X(f"{delta_col}:Q", title="Δ Year N→N+1 (kBtu/sq ft)"),
                y=alt.Y(f"{delta_next_col}:Q", title="Δ Year N+1→N+2 (kBtu/sq ft)"),
                color=alt.condition(type_select, f"{property_col}:N", alt.value("lightgray")),
                opacity=alt.condition(type_select, alt.value(0.8), alt.value(0.15)),
                tooltip=[
                    id_col,
                    property_col,
                    year_col,
                    alt.Tooltip(f"{delta_col}:Q", format=".2f"),
                    alt.Tooltip(f"{delta_next_col}:Q", format=".2f"),
                ],
            )
            .add_params(type_select)
            .properties(width=width, height=height, title=str(year))
        )

        reg = (
            alt.Chart(df_year)
            .transform_regression(delta_col, delta_next_col, groupby=[property_col])
            .mark_line(size=2)
            .encode(
                x=f"{delta_col}:Q",
                y=f"{delta_next_col}:Q",
                color=f"{property_col}:N",
                opacity=alt.condition(type_select, alt.value(1), alt.value(0.2)),
            )
        )

        return scatter + reg

    # Build year grid (same as before)
    grid_years = [
        years[0:3],
        years[3:5],
    ]
    row_n = 3
    for row in grid_years:
        while len(row) < row_n:
            row.append(None)

    rows = []
    for row_years in grid_years:
        charts = [
            make_chart(y)
            if y is not None
            else alt.Chart().mark_text(text="").properties(width=width, height=height)
            for y in row_years
        ]
        rows.append(
            alt.hconcat(*charts).resolve_scale(x="shared", y="shared")
        )

    return rows


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
    dff: pd.DataFrame, geojson: dict, metrics: list[str]
) -> alt.Chart:
    """Generate choropleth grids for all specified metrics and return the combined chart.

    Each metric produces one combined grid with an all-year average map and yearly maps.
    """
    grid_cols = 4
    combined_charts = {}

    for metric in metrics:
        safe_metric = re.sub(r"[^A-Za-z0-9_]", "_", metric).strip("_")
        agg = aggregate_metric(dff, metric)
        charts = [plot_choropleth(geojson, agg, metric)]

        for year in sorted(dff["Data Year"].dropna().unique()):
            charts.append(plot_choropleth(geojson, agg, metric, int(year)))

        grid = []
        for i in range(0, len(charts), grid_cols):
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

        combined_charts[safe_metric] = combined

    # If only one metric, return the chart directly
    if len(metrics) == 1:
        return combined_charts[safe_metric]
    return combined_charts


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
