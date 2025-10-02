import geopandas as gpd
import pandas as pd

from features import buffer


def add_h3_embeddings(buildings: gpd.GeoDataFrame, satclip_path: str) -> gpd.GeoDataFrame:
    """
    Merges precomputed SatCLIP embeddings into the buildings GeoDataFrame based on H3 index.

    Args:
        buildings: A GeoDataFrame containing building geometries.
        satclip_path: The file path to the SatCLIP embeddings data.

    Returns:
        A GeoDataFrame with the merged SatCLIP embeddings.
    """
    embeddings = pd.read_parquet(satclip_path)
    buildings['h3_08'] = buffer.h3_index(buildings, 8)
    buildings = buildings.merge(embeddings, left_on='h3_08', right_index=True, how="left")

    return buildings
