from typing import Union, List, Dict

import pandas as pd
import geopandas as gpd
from shapely.geometry import box


def sjoin_nearest_cols(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, cols: Union[List[str], Dict[str, str]], distance_col: str = None, max_distance: float = None) -> gpd.GeoDataFrame:
    if isinstance(cols, dict):
        gdf2 = gdf2.rename(columns=cols)
        cols = list(cols.values())

    gdf1 = gdf1.sjoin_nearest(gdf2[['geometry'] + cols], how='left', distance_col=distance_col, max_distance=max_distance)
    gdf1 = gdf1.drop(columns='index_right')
    gdf1 = gdf1[~gdf1.index.duplicated()]
    if distance_col:
        gdf1[distance_col] = gdf1[distance_col].fillna(max_distance)

    return gdf1


def snearest_attr(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, attr: str, max_distance: float = None) -> pd.DataFrame:
    right = right.dropna(subset=attr)

    (left_i, right_i), dis = right.sindex.nearest(
        left.geometry, return_all=False, return_distance=True, max_distance=max_distance
    )

    nearest = right.iloc[right_i][attr].reset_index()
    nearest.index = left.index[left_i]
    nearest['distance'] = dis

    return nearest


def snearest(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, max_distance: float = None) -> pd.DataFrame:
    (left_i, right_i), dis = right.sindex.nearest(
        left.geometry, return_all=False, return_distance=True, max_distance=max_distance
    )

    nearest = right.iloc[right_i].reset_index()
    nearest.index = left.index[left_i]
    nearest['distance'] = dis

    return nearest


def distance_nearest(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, max_distance: float = None) -> pd.DataFrame:
    (left_i, _), dis = right.sindex.nearest(
        left.geometry, return_all=False, return_distance=True, max_distance=max_distance
    )

    s = pd.Series(None, index=left.index, name='distance')
    s.iloc[left_i] = dis

    return s


def bbox(geom: Union[gpd.GeoSeries, gpd.GeoDataFrame], buffer: float = 0) -> gpd.GeoSeries:
    bounds = geom.total_bounds
    buffered_box = box(*bounds).buffer(buffer)
    return gpd.GeoSeries([buffered_box], crs=geom.crs)
