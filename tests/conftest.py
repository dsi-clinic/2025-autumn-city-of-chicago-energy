"""Pytest configuration and shared fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_energy_df():
    """Create a sample energy dataframe for testing."""
    return pd.DataFrame({
        "ID": ["A", "A", "A", "B", "B", "B"],
        "Data Year": [2018, 2019, 2020, 2018, 2019, 2020],
        "Primary Property Type": ["Office", "Office", "Office", "Retail", "Retail", "Retail"],
        "Community Area": ["Loop", "Loop", "Loop", "Hyde Park", "Hyde Park", "Hyde Park"],
        "ENERGY STAR Score": [80, 85, 90, 70, 75, 80],
        "Site EUI (kBtu/sq ft)": [100, 95, 90, 120, 115, 110],
        "Year Built": [1980, 1980, 1980, 1990, 1990, 1990],
        "Reporting Status": ["submitted"] * 6,
        "Latitude": [41.8, 41.8, 41.8, 41.8, 41.8, 41.8],
        "Longitude": [-87.6, -87.6, -87.6, -87.7, -87.7, -87.7]
    })


@pytest.fixture
def sample_compliance_df():
    """Create a sample dataframe with compliance flags."""
    return pd.DataFrame({
        "ID": ["A", "B", "C", "D"],
        "Data Year": [2020] * 4,
        "Time Built": ["Pre-1945", "Pre-1945", "1945-1969", "1945-1969"],
        "Primary Property Type": ["Office", "Office", "Retail", "Retail"],
        "Community Area": ["Loop", "Loop", "Hyde Park", "Hyde Park"],
        "SubmittedFlag": [1, 0, 1, 1],
        "ExemptFlag": [0, 0, 0, 0],
        "NotSubmittedFlag": [0, 1, 0, 0],
        "NonCompliantFlag": [0, 1, 0, 1]
    })


@pytest.fixture
def sample_geojson():
    """Create a minimal GeoJSON for testing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "area_numbe": "1",
                    "community": "Loop"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-87.6, 41.8],
                        [-87.6, 41.9],
                        [-87.5, 41.9],
                        [-87.5, 41.8],
                        [-87.6, 41.8]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "area_numbe": "2",
                    "community": "Hyde Park"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-87.7, 41.8],
                        [-87.7, 41.9],
                        [-87.6, 41.9],
                        [-87.6, 41.8],
                        [-87.7, 41.8]
                    ]]
                }
            }
        ]
    }
