"""Integration tests for end-to-end workflows."""

import pandas as pd

from utils.dashboard_utils import (
    aggregate_compliance_over_time,
    filter_energy_by_selections,
)
from utils.data_utils import (
    add_compliance_status,
    add_top_level_property_type,
    assign_effective_year_built,
    categorize_time_built,
    clean_property_type,
    clean_year_built,
)


class TestDataPipeline:
    """Test the full data cleaning and preparation pipeline."""

    def test_full_cleaning_pipeline(self):
        """Test that data can go through all cleaning steps without errors."""
        # Create raw data
        df = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "Data Year": [2020, 2020, 2020],
                "Primary Property Type": ["office", "retail store", "hotel"],
                "Year Built": [1950.0, 1980.0, 2000.0],
                "Community Area": ["loop", "hyde park", "loop"],
                "ENERGY STAR Score": [80, 70, 90],
                "Reporting Status": ["submitted", "submitted", "not submitted"],
                "Electricity Use (kBtu)": [1000, 2000, None],
                "Natural Gas Use (kBtu)": [500, 1000, None],
            }
        )

        # Apply cleaning steps in order
        df = clean_year_built(df)
        df = assign_effective_year_built(df)
        df = clean_property_type(df)
        df = categorize_time_built(df)
        df = add_top_level_property_type(df)

        energy_cols = ["Electricity Use (kBtu)", "Natural Gas Use (kBtu)"]
        df = add_compliance_status(df, energy_cols=energy_cols)

        # Verify all expected columns exist
        assert "Time Built" in df.columns
        assert "Top Level Property Type" in df.columns
        assert "SubmittedFlag" in df.columns
        assert "NonCompliantFlag" in df.columns

        # Verify data integrity
        assert not df.empty
        assert all(df["Year Built"].notna())
        assert all(df["Primary Property Type"].notna())

    def test_filtering_after_cleaning(self):
        """Test that filtering works correctly after cleaning."""
        df = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "Data Year": [2020, 2020, 2020],
                "Primary Property Type": ["Office", "Retail", "Office"],
                "Year Built": [1940, 1960, 1980],
                "Community Area": ["Loop", "Hyde Park", "Loop"],
                "Top Level Property Type": ["Commercial", "Commercial", "Commercial"],
            }
        )

        # Apply categorization
        df = categorize_time_built(df)

        # Test filtering
        filtered = filter_energy_by_selections(
            df,
            sel_time_built=["Pre-1945"],
            sel_ppt=["Office"],
            sel_ca=["Loop"],
            sel_tlpt=["Commercial"],
        )

        assert len(filtered) == 1
        assert filtered["ID"].iloc[0] == "A"

    def test_compliance_aggregation_pipeline(self):
        """Test compliance aggregation after data preparation."""
        df = pd.DataFrame(
            {
                "ID": ["A", "B", "C", "D"],
                "Data Year": [2020] * 4,
                "Primary Property Type": ["Office"] * 4,
                "Year Built": [1940, 1960, 1980, 2000],
                "Community Area": ["Loop"] * 4,
                "Reporting Status": [
                    "submitted",
                    "submitted",
                    "not submitted",
                    "exempt",
                ],
                "Electricity Use (kBtu)": [1000, 2000, None, 1500],
                "Natural Gas Use (kBtu)": [500, 1000, None, 750],
            }
        )

        # Prepare data
        df = categorize_time_built(df)
        energy_cols = ["Electricity Use (kBtu)", "Natural Gas Use (kBtu)"]
        df = add_compliance_status(df, energy_cols=energy_cols)

        # Aggregate compliance
        agg = aggregate_compliance_over_time(df, category_col="Time Built")

        assert not agg.empty
        assert "n_buildings" in agg.columns
        assert "share_submitted" in agg.columns
        assert all(agg["n_buildings"] > 0)


class TestDataConsistency:
    """Test data consistency across transformations."""

    def test_id_preservation(self):
        """Test that building IDs are preserved through transformations."""
        df = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "Data Year": [2020, 2020, 2020],
                "Primary Property Type": ["office", "retail", "hotel"],
                "Year Built": [1950, 1980, 2000],
            }
        )

        original_ids = set(df["ID"])

        # Apply transformations
        df = clean_property_type(df)
        df = clean_year_built(df)

        # IDs should be unchanged
        assert set(df["ID"]) == original_ids

    def test_row_count_preservation(self):
        """Test that row counts are preserved (unless intentionally filtered)."""
        df = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "Data Year": [2020, 2020, 2020],
                "Primary Property Type": ["office", "retail", "hotel"],
                "Year Built": [1950, 1980, 2000],
            }
        )

        original_len = len(df)

        # These operations should not drop rows
        df = clean_year_built(df)
        df = assign_effective_year_built(df)
        df = categorize_time_built(df)

        assert len(df) == original_len

    def test_data_type_consistency(self):
        """Test that data types are consistent after transformations."""
        df = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "Data Year": [2020, 2020, 2020],
                "Primary Property Type": ["office", "retail", "hotel"],
                "Year Built": [1950.0, 1980.0, 2000.0],
            }
        )

        df = clean_year_built(df)

        # Year Built should be integer type
        assert df["Year Built"].dtype == "Int64"

        df = clean_property_type(df)

        # Primary Property Type should be string/object
        assert df["Primary Property Type"].dtype == object


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_dataframe_handling(self):
        """Test that functions handle empty dataframes gracefully."""
        df = pd.DataFrame(
            {"ID": [], "Data Year": [], "Primary Property Type": [], "Year Built": []}
        )

        # Should not raise errors
        df = clean_year_built(df)
        df = categorize_time_built(df)

        assert len(df) == 0

    def test_all_nan_handling(self):
        """Test handling of columns with all NaN values."""
        df = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "Data Year": [2020, 2020, 2020],
                "Primary Property Type": ["Office", "Retail", "Hotel"],
                "Year Built": [None, None, None],
            }
        )

        df = clean_year_built(df)
        df = categorize_time_built(df)

        # Should complete without errors
        assert "Time Built" in df.columns
        assert all(df["Time Built"].isna())

    def test_invalid_year_handling(self):
        """Test handling of invalid year values."""
        df = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "Year Built": [1700, 1950, 2100],  # Too old, valid, too new
            }
        )

        df = clean_year_built(df)

        assert pd.isna(df["Year Built"].iloc[0])  # Too old
        assert df["Year Built"].iloc[1] == 1950  # Valid
        assert pd.isna(df["Year Built"].iloc[2])  # Too new
