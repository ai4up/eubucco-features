import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon

import util


def calculate_phi(buildings: gpd.GeoDataFrame) -> pd.Series:
    max_dist = buildings.geometry.map(lambda g: g.centroid.hausdorff_distance(g.exterior))
    circle_area = buildings.centroid.buffer(max_dist).area
    return buildings.area / circle_area


def calculate_touches(buildings: gpd.GeoDataFrame) -> pd.Series:
    touching_pairs = gpd.sjoin(buildings, buildings, predicate="intersects")
    touches = touching_pairs.groupby("id_left").size() - 1
    return buildings["id"].map(touches).fillna(0).astype(int)


def calculate_norm_perimeter(buildings: gpd.GeoDataFrame) -> pd.Series:
    return _circle_perimeter(buildings.area) / buildings.length


def _circle_perimeter(area: pd.Series) -> pd.Series:
    return 2 * np.sqrt(area * math.pi)


def calculate_distance_to_closest_building(buildings: gpd.GeoDataFrame) -> pd.Series:
    return util.distance_nearest(buildings, buildings, max_distance=100)


def remove_buildings_with_multiple_parts(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep = []
    for _, bldg in buildings.iterrows():
        geom = bldg.geometry

        if isinstance(geom, MultiPolygon):
            geom = shapely.unary_union(geom)

        if isinstance(geom, Polygon):
            keep.append(bldg)

    print(f"Removed {len(buildings) - len(keep)} buildings consisting of multiple, disconntected parts.")

    return gpd.GeoDataFrame(keep, crs=buildings.crs)
