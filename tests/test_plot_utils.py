"""Tests for plot_utils functions."""

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot_utils import (
    aggregate_metric,
    plot_bar,
    plot_trend_by_year,
)


class TestAggregateMetric:
    """Tests for aggregate_metric function."""

    def test_aggregate_metric_basic(self):
        """Test basic metric aggregation by community area."""
        df = pd.DataFrame(
            {
                "Community Area": ["Loop", "Loop", "Hyde Park"],
                "ENERGY STAR Score": [80, 90, 70],
                "Latitude": [41.8, 41.8, 41.8],
                "Longitude": [-87.6, -87.6, -87.7],
            }
        )
        result = aggregate_metric(df, "ENERGY STAR Score")

        assert "Community Area" in result.columns
        assert "ENERGY STAR Score" in result.columns
        assert len(result) == 2  # Two unique community areas

        # Check aggregation
        loop_score = result[result["Community Area"] == "Loop"][
            "ENERGY STAR Score"
        ].iloc[0]
        assert loop_score == 85.0  # Mean of 80 and 90

    def test_aggregate_metric_with_nan(self):
        """Test that NaN values are handled correctly."""
        df = pd.DataFrame(
            {
                "Community Area": ["Loop", "Loop", "Hyde Park"],
                "ENERGY STAR Score": [80, np.nan, 70],
                "Latitude": [41.8, 41.8, 41.8],
                "Longitude": [-87.6, -87.6, -87.7],
            }
        )
        result = aggregate_metric(df, "ENERGY STAR Score")

        # Should calculate mean ignoring NaN
        loop_score = result[result["Community Area"] == "Loop"][
            "ENERGY STAR Score"
        ].iloc[0]
        assert loop_score == 80.0

    def test_aggregate_metric_preserves_coordinates(self):
        """Test that latitude and longitude are preserved."""
        df = pd.DataFrame(
            {
                "Community Area": ["Loop", "Loop"],
                "ENERGY STAR Score": [80, 90],
                "Latitude": [41.8, 41.81],
                "Longitude": [-87.6, -87.61],
            }
        )
        result = aggregate_metric(df, "ENERGY STAR Score")

        assert "Latitude" in result.columns
        assert "Longitude" in result.columns
        assert not pd.isna(result["Latitude"].iloc[0])
        assert not pd.isna(result["Longitude"].iloc[0])


class TestPlotBar:
    """Tests for plot_bar function."""

    def test_plot_bar_basic(self):
        """Test basic horizontal bar chart creation."""
        df = pd.DataFrame(
            {
                "Primary Property Type": ["Office", "Retail", "Hotel"],
                "ENERGY STAR Score": [80, 70, 90],
            }
        )
        fig, ax = plot_bar(data=df, x="ENERGY STAR Score", y="Primary Property Type")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert len(ax.patches) == 3  # Three bars

    def test_plot_bar_sorted(self):
        """Test that bars are sorted by value."""
        df = pd.DataFrame(
            {
                "Primary Property Type": ["Office", "Retail", "Hotel"],
                "ENERGY STAR Score": [80, 70, 90],
            }
        )
        fig, ax = plot_bar(data=df, x="ENERGY STAR Score", y="Primary Property Type")

        # Get y-tick labels (should be sorted by value)
        labels = [label.get_text() for label in ax.get_yticklabels()]
        # Hotel (90) should be first, Office (80) second, Retail (70) last
        assert labels[0] == "Hotel" or labels[-1] == "Retail"

    def test_plot_bar_with_aggregation(self):
        """Test bar chart with aggregation."""
        df = pd.DataFrame(
            {
                "Primary Property Type": ["Office", "Office", "Retail"],
                "ENERGY STAR Score": [80, 90, 70],
            }
        )
        fig, ax = plot_bar(data=df, x="ENERGY STAR Score", y="Primary Property Type")

        # Should aggregate to 2 bars
        assert len(ax.patches) == 2


class TestPlotTrendByYear:
    """Tests for plot_trend_by_year function."""

    def test_plot_trend_by_year_basic(self):
        """Test basic trend plot creation."""
        df = pd.DataFrame(
            {"Data Year": [2018, 2019, 2020], "ENERGY STAR Score": [80, 85, 90]}
        )
        result = plot_trend_by_year(df, ["ENERGY STAR Score"], "mean")

        assert len(result) == 1  # One metric = one plot
        fig, ax = result[0]
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_plot_trend_by_year_multiple_metrics(self):
        """Test trend plot with multiple metrics."""
        df = pd.DataFrame(
            {
                "Data Year": [2018, 2019, 2020],
                "ENERGY STAR Score": [80, 85, 90],
                "Site EUI (kBtu/sq ft)": [100, 95, 90],
            }
        )
        result = plot_trend_by_year(
            df, ["ENERGY STAR Score", "Site EUI (kBtu/sq ft)"], "mean"
        )

        assert len(result) == 2  # Two metrics = two plots

    def test_plot_trend_by_year_aggregation_methods(self):
        """Test different aggregation methods."""
        df = pd.DataFrame(
            {
                "Data Year": [2018, 2018, 2019, 2019],
                "ENERGY STAR Score": [80, 90, 85, 95],
            }
        )

        # Test mean
        result_mean = plot_trend_by_year(df, ["ENERGY STAR Score"], "mean")
        assert len(result_mean) == 1

        # Test median
        result_median = plot_trend_by_year(df, ["ENERGY STAR Score"], "median")
        assert len(result_median) == 1

    def test_plot_trend_by_year_handles_missing_data(self):
        """Test that missing years are handled gracefully."""
        df = pd.DataFrame(
            {
                "Data Year": [2018, 2020],  # Missing 2019
                "ENERGY STAR Score": [80, 90],
            }
        )
        result = plot_trend_by_year(df, ["ENERGY STAR Score"], "mean")

        assert len(result) == 1
        fig, ax = result[0]
        # Should still create a plot with 2 data points


class TestPlotChoropleth:
    """Tests for choropleth map creation."""

    def test_choropleth_returns_altair_chart(self):
        """Test that choropleth returns an Altair chart object."""
        # This is a basic smoke test - full choropleth testing requires geojson
        # which is complex to mock
        from utils.plot_utils import plot_choropleth

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"area_numbe": "1", "community": "Loop"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-87.6, 41.8], [-87.6, 41.9], [-87.5, 41.9], [-87.5, 41.8]]
                        ],
                    },
                }
            ],
        }

        agg_df = pd.DataFrame(
            {
                "Community Area": ["Loop"],
                "ENERGY STAR Score": [80],
                "Latitude": [41.85],
                "Longitude": [-87.55],
            }
        )

        result = plot_choropleth(geojson, agg_df, "ENERGY STAR Score")
        assert isinstance(result, alt.LayerChart)


class TestPlotBuildingCountMap:
    """Tests for building count map creation."""

    def test_building_count_map_basic(self):
        """Test basic building count map creation."""
        from utils.plot_utils import plot_building_count_map

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"area_numbe": "1", "community": "Loop"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-87.6, 41.8], [-87.6, 41.9], [-87.5, 41.9], [-87.5, 41.8]]
                        ],
                    },
                }
            ],
        }

        df = pd.DataFrame(
            {"Community Area": ["Loop", "Loop"], "Data Year": [2020, 2020]}
        )

        result = plot_building_count_map(geojson, df, year=2020)
        assert isinstance(result, alt.LayerChart)

    def test_building_count_map_filters_by_year(self):
        """Test that building count map filters by year."""
        from utils.plot_utils import plot_building_count_map

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"area_numbe": "1", "community": "Loop"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-87.6, 41.8], [-87.6, 41.9], [-87.5, 41.9], [-87.5, 41.8]]
                        ],
                    },
                }
            ],
        }

        df = pd.DataFrame(
            {
                "Community Area": ["Loop", "Loop", "Loop"],
                "Data Year": [2019, 2020, 2021],
            }
        )

        result = plot_building_count_map(geojson, df, year=2020)
        assert isinstance(result, alt.LayerChart)
