import numpy as np
import geopandas as gpd
import pandas as pd
import shapely

from util import bbox, transform_crs, distance_nearest

# CORINE landuse classes https://land.copernicus.eu/en/products/corine-land-cover
CORINE_LU_CLASS_COL = "Code_18"
CORINE_LU_CLASSES = {"agricultural": [211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244], "industrial": [121]}
CORINE_CRS = "EPSG:3035"
OCEANS_CRS = "EPSG:3857"


def distance_to_landuse(buildings: gpd.GeoDataFrame, category: str, landuse_path: str) -> pd.Series:
    """
    Calculate the distance from each building to the nearest land use area of a specified category.

    Args:
        buildings: A GeoDataFrame containing building geometries.
        category: The land use category to calculate distances to.
        landuse_path: The file path to the land use data.

    Returns:
        A Series containing the distances from each building to the nearest land use area of the specified category.
    """
    box = bbox(buildings, crs=CORINE_CRS, buffer=1000)
    lu = gpd.read_file(landuse_path, bbox=box)
    lu = lu[lu[CORINE_LU_CLASS_COL].astype(int).isin(CORINE_LU_CLASSES[category])]
    lu = lu.to_crs(buildings.crs)
    lu = _tile_landuse(lu)

    dis = distance_nearest(buildings.centroid, lu, max_distance=1000)

    return dis


def distance_to_coast(buildings: gpd.GeoDataFrame, oceans_path: str) -> pd.Series:
    """
    Calculate the distance from each building to the nearest ocean or sea.

    Args:
        buildings: A GeoDataFrame containing building geometries.
        oceans_path: The file path to the oceans and seas data.

    Returns:
        A Series containing the distances from each building to the nearest ocean or sea.
    """
    box = bbox(buildings, crs=OCEANS_CRS, buffer=1e6)
    oceans = gpd.read_file(oceans_path, bbox=box)

    ocean_geom = oceans.union_all()
    ocean_geom = transform_crs(ocean_geom, oceans.crs, buildings.crs)
    ocean_geom_rough = ocean_geom.simplify(1000)

    approx_dis = buildings.centroid.distance(ocean_geom_rough)

    near_mask = approx_dis < 10000
    if near_mask.any():
        approx_dis.loc[near_mask] = buildings.loc[near_mask].centroid.distance(ocean_geom)

    return buildings.centroid.distance(ocean_geom)


def _tile_landuse(lu: gpd.GeoDataFrame, tile_size: int = 1000) -> gpd.GeoDataFrame:
    bounds = lu.total_bounds
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(xmin, xmax + tile_size, tile_size)
    ys = np.arange(ymin, ymax + tile_size, tile_size)
    tiles = [shapely.box(x, y, x + tile_size, y + tile_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=tiles, crs=lu.crs)

    # Intersect polygons with grid
    lu_tiled = gpd.overlay(lu, grid, how="intersection")

    return lu_tiled
