"""Utility functions for loading Chicago Energy Benchmarking datasets and GeoJSON files."""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


# will be replaced after refactoring
def load_energy_datasets(data_dir: Path) -> pd.DataFrame:
    """Loads all Chicago Energy Benchmarking CSV files from the specified directory.

    Args:
        data_dir: Path to the directory containing energy benchmarking CSV files.

    Returns:
        Combined pandas DataFrame containing all loaded CSV data.

    Raises:
        ValueError: If no CSV files are found in the specified directory.
    """
    data_dir = Path(data_dir)
    logger.info(f"Searching for CSV files under: {data_dir.resolve()}")

    csv_files = list(data_dir.glob("*.csv"))  # Non-recursive
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir.resolve()}")

    datasets = {}
    for csv_file in csv_files:
        try:
            logger.info(f"Loading: {csv_file.name}")
            csv_data = pd.read_csv(csv_file)
            datasets[csv_file.name] = csv_data
        except Exception as e:
            logger.warning(f"Could not load {csv_file.name}: {e}")

    combined_df = pd.concat(datasets.values(), ignore_index=True)
    logger.info(f"Loaded {len(datasets)} files — combined shape: {combined_df.shape}")

    if "Data Year" in combined_df.columns:
        years = sorted(combined_df["Data Year"].dropna().unique())
        logger.info(f"Years included: {years}")

    return combined_df


def load_neighborhood_geojson(geojson_path: Path) -> dict:
    """Loads a neighborhood GeoJSON file.

    Args:
        geojson_path: Path to the neighborhood GeoJSON file.

    Returns:
        A Python dictionary parsed from the GeoJSON file.
    """
    geojson_path = Path(geojson_path)
    logger.info(f"Loading GeoJSON from: {geojson_path.resolve()}")
    with geojson_path.open() as f:
        geojson = json.load(f)

    logger.info(f"Loaded {len(geojson['features'])} features")
    return geojson
