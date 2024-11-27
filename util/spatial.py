from typing import Union, List, Dict

import geopandas as gpd

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
