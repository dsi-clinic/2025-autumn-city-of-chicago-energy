"""fix_effect_utils.py

Provides functions to generate descriptive statistics and missing-value summaries
for the Chicago Energy Benchmarking dataset.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from utils.data_utils import (
    assign_effective_year_built,
    categorize_time_built,
    clean_property_type,
    concurrent_buildings,
    covid_impact_category,
)

logger = logging.getLogger(__name__)

PVAL_THRESHOLDS = [0.01, 0.05, 0.1]


CORE_COLS = [
    "Data Year",
    "Chicago Energy Rating",
    "Exempt From Chicago Energy Rating",
    "ENERGY STAR Score",
    "Site EUI (kBtu/sq ft)",
    "Source EUI (kBtu/sq ft)",
    "Weather Normalized Site EUI (kBtu/sq ft)",
    "Weather Normalized Source EUI (kBtu/sq ft)",
    "Total GHG Emissions (Metric Tons CO2e)",
    "GHG Intensity (kg CO2e/sq ft)",
    "Gross Floor Area - Buildings (sq ft)",
    "Year Built",
    "# of Buildings",
    "Primary Property Type",
    "Water Use (kGal)",
    "Electricity Use (kBtu)",
    "Natural Gas Use (kBtu)",
    "District Steam Use (kBtu)",
    "District Chilled Water Use (kBtu)",
    "All Other Fuel Use (kBtu)",
]

POLICY_YEAR = 2019
LOW_RATING_THRESHOLD = 2
PVAL_THRESHOLDS = (0.01, 0.05, 0.1)


def prepared_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Load and clean data

    Using concurrent_buildings to ensure longitudinal history for each building to allow for Fixed Effects.
    df: The object of the loaded in dataframe (planned for load_data())

    pd.DataFrame: A dataframe that is cleaned and adds columns to standardize categories
    """
    df_clean = concurrent_buildings(df)
    df_panel = clean_property_type(df_clean)
    df_panel = covid_impact_category(df_panel)
    df_panel = assign_effective_year_built(df_panel)
    df_panel = categorize_time_built(df_panel)

    return df_panel


def prepare_analysis(
    df_panel: pd.DataFrame,
    target_col: str,
    rating_col: str,
    year: int = POLICY_YEAR,
    lower_cutoff: float = 0.01,
    upper_cutoff: float = 0.99,
) -> pd.DataFrame:
    """For predictive analysis paired with the fixed_effects_analysis.

    df_panel: Base dataframe that will be prepared for longitudinal fixed effects analysis
    target_col: Chosen colummn that will manipulated for analysis
    rating_col: Chosen column for 'Rating_Cat' (String) for modeling and 'Rating_Numeric' (Float) for sorting
    year: The year of interest that will filter everything beforehand
    lower_cutoff: The lower quantile used in `Future_Change_Raw`to remove any rows below this quantile
    Helps reduce the influence of unusually large drops.
    upper_cutoff: The upper quantile used in `Future_Change_Raw`to remove any rows aboce this quantile
    Helps reduce the influence of unusually large drops.

    pd.DataFrame: Cleaned dataframe to show change year-over-year (Future_Change) for fixed effects modeling
    """
    initial_count = len(df_panel)
    initial_unique_buildings = df_panel["ID"].nunique()

    df_panel = df_panel.sort_values(by=["ID", "Data Year"]).copy()

    df_panel["Current_EUI"] = df_panel[target_col]
    df_panel["Next_Year_EUI"] = df_panel.groupby("ID")[target_col].shift(-1)
    df_panel["Future_Change_Raw"] = df_panel["Next_Year_EUI"] - df_panel["Current_EUI"]

    valid_mask = df_panel["Future_Change_Raw"].notna()
    rows_with_target = valid_mask.sum()
    surviving_unique_buildings = df_panel.loc[valid_mask, "ID"].nunique()
    buildings_lost = initial_unique_buildings - surviving_unique_buildings

    if rows_with_target > 0:
        lower_bound = df_panel["Future_Change_Raw"].quantile(lower_cutoff)
        upper_bound = df_panel["Future_Change_Raw"].quantile(upper_cutoff)

        df_panel["Future_Change"] = df_panel["Future_Change_Raw"].clip(
            lower=lower_bound, upper=upper_bound
        )
    else:
        df_panel["Future_Change"] = df_panel["Future_Change_Raw"]

    df_panel["Rating_Numeric"] = pd.to_numeric(df_panel[rating_col], errors="coerce")

    df_panel["Rating_Cat"] = df_panel["Rating_Numeric"].apply(
        lambda x: str(x) if pd.notna(x) else None
    )

    df_panel["Post_Policy"] = (df_panel["Data Year"] >= year).astype(int)

    print("-" * 40)
    print("DATA ANALYSIS PREP SUMMARY")
    print("-" * 40)
    print(f"Initial Total Rows:        {initial_count}")
    print(f"Initial Unique Buildings:  {initial_unique_buildings}")
    print("-" * 40)
    print(f"Rows with Future Change:   {rows_with_target}")
    print(f"Unique Buildings Retained: {surviving_unique_buildings}")
    print(f"Unique Buildings Lost: {buildings_lost}")
    print("-" * 40)

    return df_panel


def extract_model_coefficients(formula_list: list) -> pd.DataFrame:
    """Helper function to extract coefficients, errors, and p-values

    specifically for Star Ratings from a list of Statsmodels results.
    Input:
    - List of tuples
        - Within tuple -> (coefficient list, 'Formula Name')
            - coefficient list example: list(model.items())[0][1]
        - tuple exampled: base = (list(sensitivity_list.items())[0][1], "Base")

    - List example: [base, with_eui, with_eui_year, with_eui_year_age]
    """
    df_list = []

    for model, formula_name in formula_list:
        rating_coefs = model.params[model.params.index.str.contains("Rating_Cat")]

        if rating_coefs.empty:
            continue

        df_params = pd.DataFrame(
            {
                "name": rating_coefs.index,
                "coef": rating_coefs.to_numpy(),
                "err": model.bse[rating_coefs.index].to_numpy(),
                "pval": model.pvalues[rating_coefs.index].to_numpy(),
                "formula": formula_name,
            }
        )
        df_list.append(df_params)

    if not df_list:
        raise ValueError(
            "No coefficients found matching 'Rating_Cat'. Check model formulas."
        )

    df_full = pd.concat(df_list, ignore_index=True)

    df_full["star_rating"] = (
        df_full["name"].str.extract(r"\[T\.([\d\.]+)\]").astype(float)
    )

    return df_full


def fixed_effects_analysis(df: pd.DataFrame) -> RegressionResultsWrapper:
    """Implements Two-Way Fixed Effects (Building & Year) with datafame from prepare_analysis function

    To analyze the impact of star ratings and Data Year

    df: The prepared dataframe made to be paired with this function for fixed-effect analysis
    RegressionResultsWrapper: The fixed effect statesmodel results
    """
    print("\n--- Model: Two-Way Fixed Effects (Building & Year) ---")

    cols_needed = ["Future_Change", "Rating_Cat", "Current_EUI", "Data Year", "ID"]

    df_fe = df.dropna(subset=cols_needed).copy()

    ref_cat = "0.0"
    print(f"Using Reference Category: {ref_cat}")

    formula = (
        f"Future_Change ~ C(Rating_Cat, Treatment(reference='{ref_cat}')) + "
        "Q('Current_EUI') + C(Q('Data Year'))"
    )

    model = smf.ols(formula, data=df_fe).fit(
        cov_type="cluster", cov_kwds={"groups": df_fe["ID"]}
    )

    print(model.summary())
    return model


def run_sensitivity_models(df: pd.DataFrame) -> RegressionResultsWrapper:
    """Runs the 4-tiered sensitivity analysis.

    Inputs:
        - df: the prepared dataframe that uses prepare_analysis()

    Returns:
        A Dictionary where:
        - Keys = Model Names (strings)
        - Values = Fitted Statsmodels Objects (RegressionResultsWrapper)
    """
    formulas = {
        "1. Base Star Rating": "Future_Change ~ C(Rating_Cat, Treatment(reference='0.0'))",
        "2. +Control (EUI)": "Future_Change ~ C(Rating_Cat, Treatment(reference='0.0')) + Q('Current_EUI')",
        "3. +Control (EUI + Year)": "Future_Change ~ C(Rating_Cat, Treatment(reference='0.0')) + Q('Current_EUI') + C(Q('Data Year'))",
        "4. +Control (EUI + Year + Age)": "Future_Change ~ C(Rating_Cat, Treatment(reference='0.0')) + Q('Current_EUI') + C(Q('Data Year')) + C(Q('Time Built'))",
    }

    fitted_models = {}

    print(f"{'Status':<20} | {'Model Name'}")
    print("-" * 40)

    for name, formula_str in formulas.items():
        try:
            # Call the Engine
            model = smf.ols(formula_str, data=df).fit(cov_type="HC1")
            print(model.summary())
            # Store the actual model object
            fitted_models[name] = model

        except Exception as e:
            print(f"{'Failed':<20} | {name} -> {e}")

    return fitted_models


def binning_rating(score: float) -> float:
    """Converts an ENERGY STAR score into a discrete Energy Rating placard bin.

    This mapping follows the Chicago Energy Rating criteria where scores
    are binned into half-point increments from 1.0 to 4.0.

    Args:
        score: The ENERGY STAR score
            Can be None or NaN.

    Returns:
        float: The placard rating (0.0 for missing, 1.0-4.0 for valid scores).
    """
    NA_RATING = 0.0
    ONE_STAR_MAX = 30
    ONE_HALF_STAR_MAX = 40
    TWO_STAR_MAX = 50
    TWO_HALF_STAR_MAX = 60
    THREE_STAR_MAX = 70
    THREE_HALF_STAR_MAX = 80
    FOUR_STAR_MAX = 100

    if pd.isna(score):
        return NA_RATING
    if score <= ONE_STAR_MAX:
        return 1.0
    if score <= ONE_HALF_STAR_MAX:
        return 1.5
    if score <= TWO_STAR_MAX:
        return 2.0
    if score <= TWO_HALF_STAR_MAX:
        return 2.5
    if score <= THREE_STAR_MAX:
        return 3.0
    if score <= THREE_HALF_STAR_MAX:
        return 3.5
    if score <= FOUR_STAR_MAX:
        return 4.0

    return NA_RATING


MAPPING_DICT = {
    "near north side": "River North",
    "south lawndale": "Little Village",
    "east garfield park": "Garfield Park",
    "west garfield park": "Garfield Park",
    "forest glen": "Sauganash,Forest Glen",
    "greater grand crossing": "Grand Crossing",
    "ohare": "O'Hare",
    "near west side": "West Loop",
}


def map_community(community: str) -> str:
    """Maps community area names to standardized neighborhood names."""
    if not isinstance(community, str):
        return str(community).title() if community is not None else ""

    community_normalized = community.lower().strip()

    if community_normalized in MAPPING_DICT:
        return MAPPING_DICT[community_normalized]

    return community.title()


def check_var_count_for_var(
    col_list: list, grouping_col: str, data: pd.DataFrame, id_col: str = "ID"
) -> None:
    """Checking how much missing data there is for columns with the g column

    col_list: list of all column names that are going to be used
    grouping_col: The variable that the groupby is done by
    """
    total_build = len(data[id_col].unique())
    for col in col_list:
        missing_count = data[col].isna().sum()
        total = len(data)
        print(f"{col}: {missing_count} missing rows ({missing_count/total:.2%})")

        index_missing = data.groupby(grouping_col)[col].apply(lambda x: x.isna().sum())
        print(f"Missing data (NA) {col} by Year:")
        if index_missing.sum() > 0:
            print(index_missing)

        gaps = data.groupby(id_col)[col].apply(lambda x: x.isna().any()).sum()
        print(
            f"# of Buildings with at least 1 missing of {col}: {gaps} ({gaps/total_build:.2%}%) buildings \n"
        )


def ols_compare_and_pred_plot(
    formula: str,
    x_var: str,
    y_var: str,
    main_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    Compare_show: bool = True,
) -> None:
    """Doing OLS regression with with y_var on x_var

    It plots the two variables to compare
    and the predicted data with the actual data in the test data
    """
    model = smf.ols(formula, data=train_data).fit()
    print(model.summary())

    test_data["Predicted_EUI"] = model.predict(test_data)
    try:
        if Compare_show:
            plot_configs = [
                (main_data, x_var, y_var, "Variable Comparison"),
                (
                    test_data,
                    "Predicted_EUI",
                    y_var,
                    f"Model Predictions with {x_var} (Predicted vs Actual)",
                ),
            ]
        else:
            plot_configs = [
                (
                    test_data,
                    "Predicted_EUI",
                    y_var,
                    f"Model Predictions with {x_var} (Predicted vs Actual)",
                ),
            ]

        for df, x_col, y_col, title_suffix in plot_configs:
            plt.figure(figsize=(8, 6))

            sns.regplot(
                data=df,
                x=x_col,
                y=y_col,
                scatter_kws={"s": 10, "alpha": 0.5},
                line_kws={"color": "blue", "linewidth": 2},
                label="Data",
            )

            ax = plt.gca()
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()

            min_val = min(x_limits[0], y_limits[0])
            max_val = max(x_limits[1], y_limits[1])

            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                color="red",
                linestyle="--",
                label="45° Identity Line",
            )

            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{title_suffix}\n(X={x_col}, Y={y_col})")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()
    except Exception as e:
        print(f"Error with feature '{x_var}': {e}")


def plotting_r_sqared(
    target_col: str,
    quantative_features: list,
    categorical_features: list,
    train_df: pd.DataFrame,
) -> None:
    """This plots the r-sqaured value of the independent variable (features) on the dependent variable (target_col)

    quantative_features example:
        quantative_features = [
            "Electricity Use (kBtu)",
            "Building_Age",
            "Site EUI (kBtu/sq ft)"]

    categorical_features example:
        categorical_features = [
            "Primary Property Type",
            "COVID Impact Category",
            "Data Year"]

    Train_df: can be any dataset
    """
    full_var = []
    for q in quantative_features:
        full_var.append(f"Q('{q}')")
    for c in categorical_features:
        full_var.append(f"C(Q('{c}'))")

    r_squared_dict = {}

    print(f"Running OLS on Target: {target_col}")
    for feature in full_var:
        try:
            formula = f"Q('{target_col}') ~ {feature}"

            model = smf.ols(formula, data=train_df).fit()

            r_squared_dict[feature] = model.rsquared
        except Exception as e:
            print(f"Error with feature '{feature}': {e}")
            r_squared_dict[feature] = 0

    sorted_r2 = sorted(r_squared_dict.items(), key=lambda x: x[1], reverse=True)
    features_sorted = [item[0] for item in sorted_r2]
    r2_values_sorted = [item[1] for item in sorted_r2]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(features_sorted, r2_values_sorted, color="skyblue")

    for bar, r2 in zip(bars, r2_values_sorted):
        width = bar.get_width()
        plt.text(
            width + 0.000005,
            bar.get_y() + bar.get_height() / 2,
            f"{r2:.3f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.xlabel("R-squared")
    plt.title(f"Feature Importance by R-squared of {target_col}")
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()
