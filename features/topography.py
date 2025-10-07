import geopandas as gpd
import pandas as pd

import util
from features import buffer


def load_elevation(elevation_file: str, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    area = util.bbox(buildings, buffer=1000)
    elevation_raster, city_meta = util.read_area(elevation_file, area)
    elevation = util.raster_to_gdf(elevation_raster[0], city_meta, point=False)
    elevation = elevation.rename(columns={"values": "elevation"})
    elevation = elevation.to_crs(buildings.crs)

    return elevation


def calculate_elevation(buildings: gpd.GeoDataFrame, elevation: gpd.GeoDataFrame) -> pd.Series:
    bldg_centroids = buildings.centroid.to_frame()
    elevation = gpd.sjoin(bldg_centroids, elevation, how="left", predicate="within")["elevation"]

    return elevation


def calculate_ruggedness(buildings: gpd.GeoDataFrame, elevation: gpd.GeoDataFrame, h3_res: int) -> pd.Series:
    h3_idx = f"h3_{h3_res}"
    buffer_fts = {"ruggedness": ("elevation", "std")}
    hex_grid = buffer.aggregate_to_h3_grid(elevation, buffer_fts, h3_res)
    buildings[h3_idx] = buffer.h3_index(buildings, h3_res)
    ruggedness = buildings.merge(hex_grid, left_on=h3_idx, right_index=True, how="left")["ruggedness"]

    return ruggedness
