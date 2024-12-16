from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pandas as pd

import util


def load_osm_buildings(buildings_dir: str, region_id: str) -> gpd.GeoDataFrame:
    # according to https://wiki.openstreetmap.org/wiki/Key:building (except for additional category for industrial)
    # fmt: off
    categories = {
        "accommodation": ["apartments", "barracks", "bungalow", "cabin", "detached", "annexe", "dormitory", "farm", "ger", "hotel", "house", "houseboat", "residential", "semidetached_house", "static_caravan", "stilt_house", "terrace", "tree_house", "trullo"],  # noqa: E501
        "commercial": ["commercial", "kiosk", "office", "retail", "supermarket", "warehouse"],
        "industrial": ["industrial"],
        "religious": ["religious", "cathedral", "chapel", "church", "kingdom_hall", "monastery", "mosque", "presbytery", "shrine", "synagogue", "temple"],  # noqa: E501
        "civic_amenity": ["bakehouse", "bridge", "civic", "college", "fire_station", "government", "gatehouse", "hospital", "kindergarten", "museum", "public", "school", "toilets", "train_station", "transportation", "university"],  # noqa: E501
        "agricultural": ["barn", "conservatory", "cowshed", "farm_auxiliary", "greenhouse", "slurry_tank", "stable", "sty", "livestock", "grandstand", "pavilion", "riding_hall", "sports_hall", "sports_centre", "stadium"],  # noqa: E501
        "storage": ["allotment_house", "boathouse", "hangar", "hut", "shed"],
        "cars": ["carport", "garage", "garages", "parking"],
        "technical_buildings": ["digester", "service", "tech_cab", "transformer_tower", "water_tower", "storage_tank", "silo"],  # noqa: E501
        "other_buildings": ["beach_hut", "bunker", "castle", "construction", "container", "guardhouse", "military", "pagoda", "quonset_hut", "roof", "ruins", "tent", "tower", "windmill", "yes"],  # noqa: E501
    }
    custom_categories = {
        "residential": categories["accommodation"],
        "industrial": categories["industrial"],
        "commercial": categories["commercial"] + categories["civic_amenity"],
        "agricultural": categories["agricultural"],
        "others": categories["other_buildings"] + categories["technical_buildings"] + categories["cars"] + categories["storage"] + categories["civic_amenity"] + categories["religious"],  # noqa: E501
    }
    # fmt: on

    custom_categories = {v: k for k, values in custom_categories.items() for v in values}
    buildings = util.load_buildings(buildings_dir, region_id)
    buildings["height"] = pd.to_numeric(buildings["height"], errors="coerce")
    buildings["type"] = buildings["type_source"].map(custom_categories)

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
