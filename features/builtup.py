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
    3: [11, 21],  # <3m
    4.5: [12, 22],  # 3-6m
    10.5: [13, 23],  # 6-15m
    22.5: [14, 24],  # 15-30m
    50: [15, 25],  # >30m
}
GHS_NDVI_CATS = {
    1: 0.15,  # low vegetation surfaces NDVI <= 0.3
    2: 0.4,  # medium vegetation surfaces 0.3 < NDVI <=0.5
    3: 0.75,  # high vegetation surfaces NDVI > 0.5
}


def load_built_up(built_up_path: str, buildings: gpd.GeoDataFrame) -> tuple[np.ndarray, dict]:
    """
    Load a cropped section of the GHS built-up raster for the area around the buildings.
    https://human-settlement.emergency.copernicus.eu/ghs_buC2023.php
    """
    area = util.bbox(buildings, crs=GHS_CRS, buffer=1000)
    data, meta = util.read_area(built_up_path, area)

    return data[0], meta


def distance_to_ghs_class(buildings: gpd.GeoDataFrame, bu_raster: np.ndarray, bu_meta: dict, category: str) -> pd.Series:
    target_classes = (GHS_USE_TYPES | GHS_HEIGHT_CATS)[category]
    mask = np.isin(bu_raster, target_classes)

    dis = util.distance_nearest_cell(buildings, bu_raster, bu_meta, mask).fillna(1_000_000)

    return dis


def ghs_height(buildings: gpd.GeoDataFrame, bu_raster: np.ndarray, bu_meta: dict) -> pd.Series:
    ghs_classes = util.read_values(buildings.centroid, bu_raster, bu_meta)
    ghs_heights = ghs_classes.map(_reverse(GHS_CAT_AVG_HEIGHTS))

    return ghs_heights.fillna(0)


def ghs_height_pooled(buildings: gpd.GeoDataFrame, bu_raster: np.ndarray, bu_meta: dict, window_size: int) -> pd.Series:
    mapping = _reverse(GHS_CAT_AVG_HEIGHTS)
    height_raster = util.map_values(bu_raster, mapping)
    ghs_heights = util.read_values_pooled(buildings.centroid, height_raster, bu_meta, window_size=window_size)

    return ghs_heights.fillna(0)


def ghs_mean_height(buildings, raster, meta, buffer_m):
    mapping = _reverse(GHS_CAT_AVG_HEIGHTS)
    height_raster = util.map_values(raster, mapping)

    return util.area_mean(buildings.centroid, height_raster, meta, buffer_m)


def ghs_mean_ndvi(buildings, raster, meta, buffer_m):
    ndvi_raster = util.map_values(raster, GHS_NDVI_CATS)

    return util.area_mean(buildings.centroid, ndvi_raster, meta, buffer_m)


def ghs_type_share(buildings, raster, meta, buffer_m, category):
    target_classes = (GHS_USE_TYPES | GHS_HEIGHT_CATS)[category]
    type_mask = np.isin(raster, target_classes).astype(np.int8)

    return util.area_mean(buildings.centroid, type_mask, meta, buffer_m)


def _reverse(d):
    return {value: key for key, values in d.items() for value in values}
