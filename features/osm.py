from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pandas as pd

import util
from features import buffer


def closest_building_type(buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = util.sjoin_nearest_cols(buildings, osm_buildings, cols=["type"], max_distance=250)
    return buildings


def closest_building_height(buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = util.sjoin_nearest_cols(buildings, osm_buildings, cols=["height"], max_distance=250)
    return buildings


def closest_building_attributes(
    buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame, attributes: Union[List[str], Dict[str, str]]
) -> gpd.GeoDataFrame:
    buildings = util.sjoin_nearest_cols(buildings, osm_buildings, cols=attributes, max_distance=250)
    return buildings


# TODO: possibly correct distance metrics by overall building coverage in region
def distance_to_some_building_type(
    buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame, use_type: str
) -> pd.Series:
    osm_buildings = osm_buildings[osm_buildings["type"] == use_type]
    dis = buildings.centroid.distance(osm_buildings.geometry.union_all())
    return dis


def distance_to_some_building_height(
    buildings: gpd.GeoDataFrame, osm_buildings: gpd.GeoDataFrame, height_range: Tuple[float, float]
) -> pd.Series:
    osm_buildings = osm_buildings[osm_buildings["height"].between(*height_range)]
    dis = buildings.centroid.distance(osm_buildings.geometry.union_all())
    return dis


def building_type_share_buffer(
    osm_buildings: gpd.GeoDataFrame, h3_res: int, k: Union[int, List[int]]
) -> gpd.GeoDataFrame:
    hex_grid_shares = buffer.calculate_h3_grid_shares(osm_buildings, "type", h3_res)
    hex_grid_shares = hex_grid_shares.unstack(level="type", fill_value=0)
    hex_grid_shares = buffer._calcuate_hex_rings_aggregate(hex_grid_shares, "mean", h3_res, k)
    return hex_grid_shares
