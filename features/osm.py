import os
from typing import List, Tuple

import geopandas as gpd
import pandas as pd

import util

module_dir = os.path.dirname(__file__)
BUILDING_TYPE_CATEGORIES_FILE = os.path.join(module_dir, "..", "data", "osm_type_matches_v1.csv")


def get_building_types() -> List[str]:
    categories = pd.read_csv(BUILDING_TYPE_CATEGORIES_FILE)
    return categories["type"].dropna().unique().tolist()


def load_osm_buildings(buildings_dir: str, region_id: str) -> gpd.GeoDataFrame:
    categories = pd.read_csv(BUILDING_TYPE_CATEGORIES_FILE)
    categories_map = categories.set_index("type_source")["type"].to_dict()

    buildings = util.load_buildings(buildings_dir, region_id)
    buildings["height"] = pd.to_numeric(buildings["height"], errors="coerce")
    buildings["type"] = buildings["type_source"].map(categories_map)

    return buildings[["geometry", "height", "type"]]


def closest_building_height(buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    nearest = util.snearest_attr(buildings, osm_buildings, attr="height", max_distance=250)
    return nearest["height"]


def closest_building_type(buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    nearest = util.snearest_attr(buildings, osm_buildings, attr="type", max_distance=250)
    nearest["type"] = pd.Categorical(nearest["type"], categories=get_building_types())

    buildings["osm_closest_building_type"] = nearest["type"]
    buildings = pd.get_dummies(buildings, columns=["osm_closest_building_type"])

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
