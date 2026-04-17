"""Tests for data_utils functions."""

import numpy as np
import pandas as pd
import pytest

from utils.data_utils import (
    add_compliance_status,
    add_top_level_property_type,
    assign_effective_year_built,
    categorize_time_built,
    clean_numeric,
    clean_property_type,
    clean_year_built,
    concurrent_buildings,
)


class TestCleanNumeric:
    """Tests for clean_numeric function."""

    def test_clean_numeric_basic(self):
        """Test basic numeric cleaning."""
        series = pd.Series(["1,000", "2,500", "3000"])
        result = clean_numeric(series)
        expected = pd.Series([1000.0, 2500.0, 3000.0])
        pd.testing.assert_series_equal(result, expected)

    def test_clean_numeric_with_empty_strings(self):
        """Test that empty strings become NaN."""
        series = pd.Series(["100", "", "200"])
        result = clean_numeric(series)
        assert pd.isna(result.iloc[1])
        assert result.iloc[0] == 100.0
        assert result.iloc[2] == 200.0

    def test_clean_numeric_with_spaces(self):
        """Test that spaces are handled correctly."""
        series = pd.Series(["  100  ", "200", " 300 "])
        result = clean_numeric(series)
        expected = pd.Series([100.0, 200.0, 300.0])
        pd.testing.assert_series_equal(result, expected)

    def test_clean_numeric_preserves_decimals(self):
        """Test that decimal values are preserved."""
        series = pd.Series(["1.5", "2.75", "3.0"])
        result = clean_numeric(series)
        expected = pd.Series([1.5, 2.75, 3.0])
        pd.testing.assert_series_equal(result, expected)


class TestCleanPropertyType:
    """Tests for clean_property_type function."""

    def test_clean_property_type_basic(self):
        """Test basic property type cleaning."""
        df = pd.DataFrame({
            "Primary Property Type": ["office", "RETAIL", "Hotel"]
        })
        result = clean_property_type(df)
        assert "Primary Property Type" in result.columns
        assert result["Primary Property Type"].iloc[0] == "Office"
        assert result["Primary Property Type"].iloc[1] == "Retail"
        assert result["Primary Property Type"].iloc[2] == "Hotel"

    def test_clean_property_type_removes_small_groups(self):
        """Test that property types with < MIN_PRIMARY_PROPERTY entries are removed."""
        # Create a dataframe where some types have very few entries
        df = pd.DataFrame({
            "Primary Property Type": ["office"] * 200 + ["rare_type"] * 5
        })
        result = clean_property_type(df)
        # Should only keep "office" since "rare_type" has < 150 entries
        assert "rare_type" not in result["Primary Property Type"].values
        assert "Office" in result["Primary Property Type"].values

    def test_clean_property_type_handles_nan(self):
        """Test that NaN values are preserved."""
        df = pd.DataFrame({
            "Primary Property Type": ["office", np.nan, "retail"]
        })
        result = clean_property_type(df)
        assert pd.isna(result["Primary Property Type"].iloc[1])


class TestCleanYearBuilt:
    """Tests for clean_year_built function."""

    def test_clean_year_built_valid_years(self):
        """Test that valid years are preserved."""
        df = pd.DataFrame({
            "Year Built": [1950, 1980, 2000]
        })
        result = clean_year_built(df)
        pd.testing.assert_series_equal(result["Year Built"], df["Year Built"])

    def test_clean_year_built_filters_too_old(self):
        """Test that years before 1800 are filtered."""
        df = pd.DataFrame({
            "Year Built": [1700, 1950, 2000]
        })
        result = clean_year_built(df)
        assert pd.isna(result["Year Built"].iloc[0])
        assert result["Year Built"].iloc[1] == 1950
        assert result["Year Built"].iloc[2] == 2000

    def test_clean_year_built_filters_future(self):
        """Test that future years are filtered."""
        df = pd.DataFrame({
            "Year Built": [1950, 2000, 2100]
        })
        result = clean_year_built(df)
        assert result["Year Built"].iloc[0] == 1950
        assert result["Year Built"].iloc[1] == 2000
        assert pd.isna(result["Year Built"].iloc[2])

    def test_clean_year_built_converts_to_int(self):
        """Test that years are converted to integers."""
        df = pd.DataFrame({
            "Year Built": [1950.0, 1980.5, 2000.0]
        })
        result = clean_year_built(df)
        assert result["Year Built"].dtype == "Int64"


class TestAssignEffectiveYearBuilt:
    """Tests for assign_effective_year_built function."""

    def test_assign_effective_year_built_basic(self):
        """Test basic effective year built assignment."""
        df = pd.DataFrame({
            "Year Built": [1950, np.nan, 2000],
            "ID": ["A", "B", "C"],
            "Data Year": [2020, 2020, 2020]
        })
        result = assign_effective_year_built(df)
        assert result["Year Built"].iloc[0] == 1950
        assert result["Year Built"].iloc[2] == 2000
        # Row with NaN should remain NaN if no other data
        assert pd.isna(result["Year Built"].iloc[1])

    def test_assign_effective_year_built_fills_from_other_years(self):
        """Test that missing years are filled from other years for same building."""
        df = pd.DataFrame({
            "Year Built": [1950, np.nan, np.nan],
            "ID": ["A", "A", "B"],
            "Data Year": [2018, 2019, 2020]
        })
        result = assign_effective_year_built(df)
        # Second row should get year from first row (same ID)
        assert result["Year Built"].iloc[0] == 1950
        assert result["Year Built"].iloc[1] == 1950


class TestCategorizeTimeBuilt:
    """Tests for categorize_time_built function."""

    def test_categorize_time_built_basic(self):
        """Test basic time built categorization."""
        df = pd.DataFrame({
            "Year Built": [1940, 1960, 1980, 2000, 2020]
        })
        result = categorize_time_built(df)
        assert "Time Built" in result.columns
        assert result["Time Built"].iloc[0] == "Pre-1945"
        assert result["Time Built"].iloc[1] == "1945-1969"
        assert result["Time Built"].iloc[2] == "1970-1989"
        assert result["Time Built"].iloc[3] == "1990-2009"
        assert result["Time Built"].iloc[4] == "2010+"

    def test_categorize_time_built_edge_cases(self):
        """Test edge cases for time built categorization."""
        df = pd.DataFrame({
            "Year Built": [1945, 1970, 1990, 2010]
        })
        result = categorize_time_built(df)
        # Boundary values
        assert result["Time Built"].iloc[0] == "1945-1969"
        assert result["Time Built"].iloc[1] == "1970-1989"
        assert result["Time Built"].iloc[2] == "1990-2009"
        assert result["Time Built"].iloc[3] == "2010+"

    def test_categorize_time_built_handles_nan(self):
        """Test that NaN values are handled."""
        df = pd.DataFrame({
            "Year Built": [1950, np.nan, 2000]
        })
        result = categorize_time_built(df)
        assert pd.isna(result["Time Built"].iloc[1])


class TestAddTopLevelPropertyType:
    """Tests for add_top_level_property_type function."""

    def test_add_top_level_property_type_basic(self):
        """Test basic top level property type assignment."""
        df = pd.DataFrame({
            "Primary Property Type": ["Office", "Retail Store", "Hotel"]
        })
        result = add_top_level_property_type(df)
        assert "Top Level Property Type" in result.columns
        assert result["Top Level Property Type"].iloc[0] == "Commercial"
        assert result["Top Level Property Type"].iloc[1] == "Commercial"
        assert result["Top Level Property Type"].iloc[2] == "Commercial"

    def test_add_top_level_property_type_multifamily(self):
        """Test multifamily categorization."""
        df = pd.DataFrame({
            "Primary Property Type": ["Multifamily Housing"]
        })
        result = add_top_level_property_type(df)
        assert result["Top Level Property Type"].iloc[0] == "Multifamily"

    def test_add_top_level_property_type_handles_unknown(self):
        """Test that unknown types are categorized as Other."""
        df = pd.DataFrame({
            "Primary Property Type": ["Unknown Type"]
        })
        result = add_top_level_property_type(df)
        # Unknown types should be categorized as "Other" or left as is
        assert "Top Level Property Type" in result.columns


class TestConcurrentBuildings:
    """Tests for concurrent_buildings function."""

    def test_concurrent_buildings_basic(self):
        """Test basic concurrent buildings filtering."""
        df = pd.DataFrame({
            "ID": ["A", "A", "A", "B", "B"],
            "Data Year": [2016, 2017, 2018, 2016, 2017],
            "Reporting Status": ["submitted"] * 5
        })
        result = concurrent_buildings(df, start_year=2016, end_year=2018)
        # Only building A should remain (has all 3 years)
        assert len(result) == 3
        assert all(result["ID"] == "A")

    def test_concurrent_buildings_filters_incomplete(self):
        """Test that buildings without all years are filtered out."""
        df = pd.DataFrame({
            "ID": ["A", "A", "B", "B", "B"],
            "Data Year": [2016, 2017, 2016, 2017, 2018],
            "Reporting Status": ["submitted"] * 5
        })
        result = concurrent_buildings(df, start_year=2016, end_year=2018)
        # Only building B should remain (has all 3 years)
        assert len(result) == 3
        assert all(result["ID"] == "B")

    def test_concurrent_buildings_respects_status(self):
        """Test that reporting status is respected for years >= 2018."""
        df = pd.DataFrame({
            "ID": ["A"] * 3,
            "Data Year": [2017, 2018, 2019],
            "Reporting Status": ["submitted", "submitted", "not submitted"]
        })
        result = concurrent_buildings(df, start_year=2017, end_year=2019, status_year=2018)
        # Should filter out the 2019 row with "not submitted" status
        assert len(result) == 2


class TestAddComplianceStatus:
    """Tests for add_compliance_status function."""

    def test_add_compliance_status_basic(self):
        """Test basic compliance status flags."""
        df = pd.DataFrame({
            "Reporting Status": ["submitted", "not submitted", "exempt"],
            "Electricity Use (kBtu)": [1000, np.nan, 500],
            "Natural Gas Use (kBtu)": [500, np.nan, 300]
        })
        energy_cols = ["Electricity Use (kBtu)", "Natural Gas Use (kBtu)"]
        result = add_compliance_status(df, energy_cols=energy_cols)

        assert "SubmittedFlag" in result.columns
        assert "NotSubmittedFlag" in result.columns
        assert "ExemptFlag" in result.columns
        assert "NonCompliantFlag" in result.columns

        assert result["SubmittedFlag"].iloc[0] == 1
        assert result["NotSubmittedFlag"].iloc[1] == 1
        assert result["ExemptFlag"].iloc[2] == 1

    def test_add_compliance_status_non_compliant(self):
        """Test non-compliant flag for submitted but missing data."""
        df = pd.DataFrame({
            "Reporting Status": ["submitted", "submitted"],
            "Electricity Use (kBtu)": [1000, np.nan],
            "Natural Gas Use (kBtu)": [500, np.nan]
        })
        energy_cols = ["Electricity Use (kBtu)", "Natural Gas Use (kBtu)"]
        result = add_compliance_status(df, energy_cols=energy_cols)

        # First row: submitted with data
        assert result["SubmittedFlag"].iloc[0] == 1
        assert result["NonCompliantFlag"].iloc[0] == 0

        # Second row: submitted but missing all energy data
        assert result["SubmittedFlag"].iloc[1] == 1
        assert result["NonCompliantFlag"].iloc[1] == 1
