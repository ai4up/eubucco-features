import geopandas as gpd
import pandas as pd

import util
from features import buffer


def calculate_elevation(buildings: gpd.GeoDataFrame, elevation_file: str) -> pd.Series:
    area = util.bbox(buildings, buffer=1000)
    elevation = util.load_elevation(elevation_file, area, point_geom=False)
    elevation = elevation.to_crs(buildings.crs)

    bldg_centroids = buildings.centroid.to_frame()
    elevation = gpd.sjoin(bldg_centroids, elevation, how='left', predicate='within')['elevation']

    return elevation


def calculate_ruggedness(buildings: gpd.GeoDataFrame, elevation_file: str, h3_res: int) -> pd.Series:
    area = util.bbox(buildings, buffer=1000)
    elevation = util.load_elevation(elevation_file, area, point_geom=True)
    elevation = elevation.to_crs(buildings.crs)

    h3_idx = f'h3_{h3_res}'
    buffer_fts = {'ruggedness': ('elevation', 'std')}
    hex_grid = buffer.aggregate_to_h3_grid(elevation, buffer_fts, h3_res)
    buildings[h3_idx] = buffer.h3_index(buildings, h3_res)
    ruggedness = buildings.merge(hex_grid, left_on=h3_idx, right_index=True, how='left')['ruggedness']

    return ruggedness
