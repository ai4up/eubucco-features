import os
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pandas as pd

import util

module_dir = os.path.dirname(__file__)
BUILDING_TYPE_CATEGORIES_FILE = os.path.join(module_dir, "..", "data", "osm_type_matches_v1.csv")


def load_osm_buildings(buildings_dir: str, region_id: str) -> gpd.GeoDataFrame:
    categories = pd.read_csv(BUILDING_TYPE_CATEGORIES_FILE)
    categories_map = categories.set_index("type_source")["type"].to_dict()

    buildings = util.load_buildings(buildings_dir, region_id)
    buildings["height"] = pd.to_numeric(buildings["height"], errors="coerce")
    buildings["type"] = buildings["type_source"].map(categories_map)

    return buildings[["geometry", "height", "type"]]


def closest_building_type(buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = util.sjoin_nearest_cols(buildings, osm_buildings, cols=["type"], max_distance=250)
    return buildings


def closest_building_height(buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = util.sjoin_nearest_cols(buildings, osm_buildings, cols=["height"], max_distance=250)
    return buildings


def closest_building_attr(
    buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame, attributes: Union[List[str], Dict[str, str]]
) -> gpd.GeoDataFrame:
    buildings = util.sjoin_nearest_cols(buildings, osm_buildings, cols=attributes, max_distance=250)
    return buildings


# TODO: possibly correct distance metrics by overall building coverage in region
def distance_to_building_type(
    buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame, use_type: str
) -> pd.Series:
    osm_buildings = osm_buildings[osm_buildings["type"] == use_type]
    dis = buildings.centroid.distance(osm_buildings.geometry.union_all())
    return dis


def distance_to_building_height(
    buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame, height_range: Tuple[float, float]
) -> pd.Series:
    osm_buildings = osm_buildings[osm_buildings["height"].between(*height_range)]
    dis = buildings.centroid.distance(osm_buildings.geometry.union_all())
    return dis
