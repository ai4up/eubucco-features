from collections import defaultdict

import geopandas as gpd
import osmnx as ox
import pandas as pd
from networkx.exception import NetworkXPointlessConcept
from shapely.geometry import Polygon

from util import distance_nearest, load_gpkg

_education = [
    "university",
    "school",
    "kindergarten",
]
_healthcare = [
    "veterinary",
    "clinic",
    "dentist",
    "pharmacy",
    "doctors",
]
_necessities = [
    "post_office",
    "fuel",
    "atm",
    "bank",
    "library",
]
_third_places = [
    "place_of_worship",
    "nightclub",
    "theatre",
    "bar",
    "cafe",
    "restaurant",
    "pub",
    "community_centre",
    "social_facility",
]
OSM_TAGS = {
    "commercial": {
        "amenity": _third_places + _necessities + _healthcare,
        "shop": True,
    },
    "industrial": {
        "industrial": True,
        "landuse": ["industrial"],
    },
    "education": {
        "amenity": _education,
    },
}


def download(area: Polygon) -> gpd.GeoDataFrame:
    """
    Downloads points of interest (POIs) from OpenStreetMap within a specified geographic area (assumes EPSG:4326).

    Args:
        area: A GeoDataFrame representing the geographic area of interest.
    Returns:
        A GeoDataFrame containing the POIs within the specified area.
    """
    try:
        ox.config(timeout=1000)
        tags = _merge_tags(*OSM_TAGS.values())
        pois = ox.geometries.geometries_from_polygon(area, tags)
        pois = pois.filter(items=["geometry"] + list(tags.keys()))

        return pois

    except NetworkXPointlessConcept:
        return None


def load_pois(pois_dir: str, region_id: str, crs: int) -> gpd.GeoDataFrame:
    pois = load_gpkg(pois_dir, region_id)
    pois = pois.to_crs(crs)

    return pois


def distance_to_closest_poi(buildings: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, category=None) -> gpd.GeoSeries:
    """
    Calculates the distance between each building and the closest Point Of Interest (POI).

    Args:
        buildings: A GeoDataFrame containing the buildings.
        pois: A GeoDataFrame containing the POIs.

    Returns:
        A GeoSeries containing the distance to the closest POI for each building.
    """
    if category:
        pois = _filter(pois, OSM_TAGS[category])

    dis = distance_nearest(buildings.centroid, pois, max_distance=1000)

    return dis


def _merge_tags(*dicts):
    merged_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            if isinstance(value, list):
                merged_dict[key].extend(value)
            elif isinstance(value, bool):
                merged_dict[key] = value
            else:
                raise ValueError(f"Merging dicts failed due to unsupported value type: {type(value)}")

    return dict(merged_dict)


def _filter(df, tags):
    mask = pd.Series(True, index=df.index)
    for col, value in tags.items():
        if isinstance(value, list):
            mask |= df[col].isin(value)
        elif value is True:
            mask |= df[col].notna()
        else:
            raise ValueError(f"Creating filter mask failed due to unsupported value type: {type(value)}")

    return df[mask]
