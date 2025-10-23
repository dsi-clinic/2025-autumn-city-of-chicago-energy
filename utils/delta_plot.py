"""Utility functions for plotting building-level energy delta and cumulative change metrics."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_building_energy_deltas(
    pivot_df: pd.DataFrame,
    metric_name: str = "Energy Use",  # <-- generalized metric label
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
