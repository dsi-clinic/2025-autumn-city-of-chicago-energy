"""visualizer.py

Provides plotting functions for the Chicago Energy Benchmarking dataset.

Includes:
- Year-grouped histograms / density plots for numeric variables
- Trend plots (mean or median by year)
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
        sharey=True,  # ensures same scale across all panels
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
