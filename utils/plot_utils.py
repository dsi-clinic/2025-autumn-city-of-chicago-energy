"""plot_utils.py

This module provides plotting utilities for visualizing energy benchmarking data.
It includes reusable functions for comparing variable distributions before and after 2019,
with support for log-scaled y-axes and clean, publication-ready visuals.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


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
