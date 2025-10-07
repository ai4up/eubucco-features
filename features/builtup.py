import geopandas as gpd
import numpy as np
import pandas as pd

import util
GHS_CRS = "ESRI:54009"
GHS_USE_TYPES = {
    "residential": [11, 12, 13, 14, 15],
    "non-residential": [21, 22, 23, 24, 25],
}
GHS_HEIGHT_CATS = {
    "low-rise": [12, 22],
    "low-medium-rise": [13, 23],
    "medium-rise": [14, 24],
    "high-rise": [15, 25],
}
GHS_CAT_AVG_HEIGHTS = {
    4.5: [12, 22],  # 3-6m
    10.5: [13, 23],  # 6-15m
    22.5: [14, 24],  # 15-30m
    50: [15, 25],  # 30m+
}
GHS_NDVI_CATS = {
    1: 0.15,  # low vegetation surfaces NDVI <= 0.3
    2: 0.4,  # medium vegetation surfaces 0.3 < NDVI <=0.5
    3: 0.75,  # high vegetation surfaces NDVI > 0.5
}


def load_built_up(landuse_path: str, buildings: gpd.GeoDataFrame) -> tuple[np.ndarray, dict]:
    area = util.bbox(buildings, crs=GHS_CRS, buffer=1000)
    data, meta = util.read_area(landuse_path, area)

    return data[0], meta

# def load_built_up(built_up_file: str, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     area = util.bbox(buildings, buffer=1000)
#     built_up_raster, city_meta = util.read_area(built_up_file, area)
#     built_up = util.raster_to_gdf(built_up_raster[0], city_meta)
#     built_up = built_up.rename(columns={"values": "class"})

#     def reverse(d):
#         return {value: key for key, values in d.items() for value in values}

#     built_up["height"] = built_up["class"].map(reverse(GHS_CAT_AVG_HEIGHTS))
#     built_up["high_rise"] = built_up["class"].isin(GHS_HIGH_RISE_CATS)
#     built_up["use_type"] = built_up["class"].map(reverse(GHS_USE_TYPES))
#     built_up["NDVI"] = built_up["class"].map(GHS_NDVI_CATS)

#     built_up = built_up.to_crs(buildings.crs)

#     return built_up


def ghs_height(buildings: gpd.GeoDataFrame, built_up_file: str) -> pd.Series:
    ghs_classes = util.read_values(built_up_file, buildings.centroid)
    ghs_heights = ghs_classes.map(_reverse(GHS_CAT_AVG_HEIGHTS))

    return ghs_heights


def ghs_type(buildings: gpd.GeoDataFrame, built_up_file: str) -> pd.Series:
    ghs_classes = util.read_values(built_up_file, buildings.centroid)
    ghs_types = ghs_classes.map(_reverse(GHS_USE_TYPES))

    return ghs_types


def distance_to_ghs_class(buildings: gpd.GeoDataFrame, bu_raster: np.ndarray, bu_meta: dict, category: str) -> pd.Series:
    target_classes = (GHS_USE_TYPES | GHS_HEIGHT_CATS)[category]
    mask = np.isin(bu_raster, target_classes)

    return util.distance_nearest_cell(buildings, bu_raster, bu_meta, mask)

def _reverse(d):
    return {value: key for key, values in d.items() for value in values}
