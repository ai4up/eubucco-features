from typing import Dict, List, Union

import geopandas as gpd
import momepy
import numpy as np
import osmnx as ox
from networkx.exception import NetworkXPointlessConcept
from shapely.geometry import Polygon

from util import distance_nearest, sjoin_nearest_cols

ROAD_SIZE: Dict[str, int] = {
    "motorway": 7,
    "trunk": 6,
    "primary": 5,
    "secondary": 4,
    "tertiary": 3,
    "unclassified": 2,
    "residential": 1,
    "living_street": 0,
    "pedestrian": 0,
}


def download(area: Polygon) -> gpd.GeoDataFrame:
    """
    Downloads streets from OpenStreetMap within a specified geographic area (assumes EPSG:4326).

    Args:
        area: A GeoDataFrame representing the geographic area of interest.
    Returns:
        A GeoDataFrame containing the POIs within the specified area.
    """
    try:
        ox.config(timeout=1000)
        street_network = ox.graph_from_polygon(area, simplify=True, network_type="drive")
        streets = ox.utils_graph.graph_to_gdfs(
            street_network, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True
        )[["osmid", "highway", "length", "geometry"]]

        return streets

    except NetworkXPointlessConcept:
        return None


def distance_to_closest_street(buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Calculates the distance between each building and the closest street.

    Args:
        buildings: A GeoDataFrame containing the buildings.
        streets: A GeoDataFrame containing the streets.

    Returns:
        A GeoSeries containing the distance to the closest street for each building.
    """
    dis = distance_nearest(buildings, streets, max_distance=100)

    return dis


def closest_street_features(buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates the size of the closest street to each building (ranging from 0 to 7, with 7 being the largest),
    the distance between the building and the closest street,
    and the deviation of the building's orientation from the street's orientation.

    Args:
        buildings: A GeoDataFrame containing the buildings.
        streets: A GeoDataFrame containing the streets.

    Returns:
        A GeoDataFrame containing the calculated size of the closest street and the distance to the closest street,
        and the orientation to the closest street for each building.
    """
    if "street_orientation" not in streets.columns:
        streets["street_orientation"] = momepy.orientation(streets)
    if "bldg_orientation" not in buildings.columns:
        buildings["bldg_orientation"] = momepy.orientation(buildings)

    streets["size"] = streets["highway"].apply(_preprocess_highway_type)
    buildings = sjoin_nearest_cols(
        buildings, streets, cols=["size", "street_orientation"], distance_col="distance", max_distance=100
    )
    buildings["street_alignment"] = (buildings["bldg_orientation"] - buildings["street_orientation"]).abs()

    return buildings[["size", "distance", "street_alignment"]]


def _preprocess_highway_type(category: Union[str, List[str]]) -> float:
    return (
        np.mean([ROAD_SIZE.get(c, np.nan) for c in category])
        if isinstance(category, list)
        else ROAD_SIZE.get(category, np.nan)
    )
