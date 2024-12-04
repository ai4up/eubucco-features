from collections import defaultdict

import geopandas as gpd
import osmnx as ox
import pandas as pd

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
    "office": {},
    "industrial": {
        "industrial": True,
        "landuse": ["industrial"],
    },
    "education": {
        "amenity": _education,
    },
}


def download(area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Downloads points of interest (POIs) within a specified geographic area.

    Args:
        area: A GeoDataFrame representing the geographic area of interest.
    Returns:
        A GeoDataFrame containing the POIs within the specified area, with the same CRS as the input area.
    """
    ox.config(timeout=1000)
    tags = _merge_tags(*OSM_TAGS.values())

    east, south, west, north = area.to_crs("EPSG:4326").total_bounds
    pois = ox.geometries.geometries_from_bbox(north, south, east, west, tags)

    pois = pois[["geometry"] + list(tags.keys())]
    pois = pois.to_crs(area.crs)

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

    any_poi = pois.geometry.union_all()
    dis = buildings.centroid.distance(any_poi)

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
            mask &= df[col].isin(value)
        elif value is True:
            mask &= df[col].notna()
        else:
            raise ValueError(f"Creating filter mask failed due to unsupported value type: {type(value)}")

    return df[mask]
