from typing import Dict, Union, List

import geopandas as gpd
import numpy as np

import util

ROAD_SIZE: Dict[str, int] = {
    'motorway': 7,
    'trunk': 6,
    'primary': 5,
    'secondary': 4,
    'tertiary': 3,
    'unclassified': 2,
    'residential': 1,
    'living_street': 0,
    'pedestrian': 0,
}

def distance_to_closest_street(buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Calculates the distance between each building and the closest street.

    Args:
        buildings: A GeoDataFrame containing the buildings.
        streets: A GeoDataFrame containing the streets.

    Returns:
        A GeoSeries containing the distance to the closest street for each building.
    """
    street_network = streets.geometry.union_all()
    dis = buildings.distance(street_network)
    return dis


def street_size_and_distance_to_closest_street(buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates the size of the closest street to each building (ranging from 0 to 7, with 7 being the largest) and the distance between the building and the closest street.

    Args:
        buildings: A GeoDataFrame containing the buildings.
        streets: A GeoDataFrame containing the streets.

    Returns:
        A GeoDataFrame with the calculated size of the closest street and the distance to the closest street for each building.
    """
    streets['size'] = streets['highway'].apply(_preprocess_highway_type)
    buildings = util.sjoin_nearest_cols(buildings, streets, cols=['size'], distance_col='distance', max_distance=100)

    return buildings[['size', 'distance']]


def _preprocess_highway_type(category: Union[str, List[str]]) -> float:
    return np.mean([ROAD_SIZE.get(c, np.nan) for c in category]) if isinstance(category, list) else ROAD_SIZE.get(category, np.nan)
