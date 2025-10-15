from typing import Any

import geopandas as gpd
import pandas as pd

import util


def closest_building(buildings: gpd.GeoDataFrame, attr: str) -> gpd.GeoDataFrame:
    nearest = util.snearest_attr(buildings, buildings, attr=attr, max_distance=50)
    return nearest[attr]


def distance_to_building(
    buildings: gpd.GeoDataFrame, attr: str, value: Any
) -> pd.Series:
    if isinstance(value, (list, tuple)) and isinstance(value[0], str):
        subset = buildings[buildings[attr].isin(value)]
    elif isinstance(value, (list, tuple)) and isinstance(value[0], (int, float)):
        subset = buildings[buildings[attr].between(*value)]
    else:
        subset = buildings[buildings[attr] == value]

    dis = util.distance_nearest(buildings, subset, max_distance=1000, exclusive=True)

    return dis.fillna(1000)
