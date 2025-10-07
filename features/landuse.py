"""
Land-use distance calculation utilities.

See https://land.copernicus.eu/en/products/corine-land-cover for
CLC codes and details on CORINE land-use classes.

Mapping between CORINE raster values, CLC codes, and descriptions:
CLC_MAPPING = [
    (1, 111, "Continuous urban fabric"),
    (2, 112, "Discontinuous urban fabric"),
    (3, 121, "Industrial or commercial units"),
    (4, 122, "Road and rail networks and associated land"),
    (5, 123, "Port areas"),
    (6, 124, "Airports"),
    (7, 131, "Mineral extraction sites"),
    (8, 132, "Dump sites"),
    (9, 133, "Construction sites"),
    (10, 141, "Green urban areas"),
    (11, 142, "Sport and leisure facilities"),
    (12, 211, "Non-irrigated arable land"),
    (13, 212, "Permanently irrigated land"),
    (14, 213, "Rice fields"),
    (15, 221, "Vineyards"),
    (16, 222, "Fruit trees and berry plantations"),
    (17, 223, "Olive groves"),
    (18, 231, "Pastures"),
    (19, 241, "Annual crops associated with permanent crops"),
    (20, 242, "Complex cultivation patterns"),
    (21, 243, "Land principally occupied by agriculture, with significant areas of natural vegetation"),
    (22, 244, "Agro-forestry areas"),
    (23, 311, "Broad-leaved forest"),
    (24, 312, "Coniferous forest"),
    (25, 313, "Mixed forest"),
    (26, 321, "Natural grasslands"),
    (27, 322, "Moors and heathland"),
    (28, 323, "Sclerophyllous vegetation"),
    (29, 324, "Transitional woodland-shrub"),
    (30, 331, "Beaches, dunes, sands"),
    (31, 332, "Bare rocks"),
    (32, 333, "Sparsely vegetated areas"),
    (33, 334, "Burnt areas"),
    (34, 335, "Glaciers and perpetual snow"),
    (35, 411, "Inland marshes"),
    (36, 412, "Peat bogs"),
    (37, 421, "Salt marshes"),
    (38, 422, "Salines"),
    (39, 423, "Intertidal flats"),
    (40, 511, "Water courses"),
    (41, 512, "Water bodies"),
    (42, 521, "Coastal lagoons"),
    (43, 522, "Estuaries"),
    (44, 523, "Sea and ocean"),
]
"""
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt

from util import bbox, transform_crs, read_area

CORINE_CRS = "EPSG:3035"
OCEANS_CRS = "EPSG:3857"

CORINE_LU_MAPPING = {
    "agricultural": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "industrial": [3],
    "dense_urban": [1],
}

def load_landuse(landuse_path: str, buildings: gpd.GeoDataFrame) -> tuple[np.ndarray, dict]:
    """
    Load a cropped section of the CORINE land-use raster for the area around the buildings.

    Args:
        landuse_path: Path to CORINE land-use GeoTIFF
        buildings: GeoDataFrame of building geometries

    Returns:
        A tuple of (landuse raster, raster metadata)
    """
    area = bbox(buildings, crs=CORINE_CRS, buffer=1000)
    data, meta = read_area(landuse_path, area)

    return data[0], meta


def distance_to_landuse(buildings: gpd.GeoDataFrame, lu_raster: np.ndarray, lu_meta: dict, category: str) -> pd.Series:
    """
    Compute approximate Euclidean distance (in meters) from each building
    to the nearest land-use grid cell of a specified CORINE category.

    The distance is calculated from the centroid of the raster cell
    containing each building to the centroid of the nearest cell that
    belongs to the given land-use category. This is an approximate measure
    based on the raster grid resolution.

    Args:
        buildings: GeoDataFrame with building geometries
        lu_raster: Land use raster data
        lu_meta: Land use raster metadata
        category: The land use category to calculate distances to.

    Returns:
        A Series containing the distances to the nearest land use area of the specified category.
    """
    target_classes = CORINE_LU_MAPPING[category]
    mask = np.isin(lu_raster, target_classes)

    if not np.any(mask):
        return pd.Series(np.nan, index=buildings.index)

    # Compute distance transform (in meters)
    px_size = lu_meta["transform"].a
    dist_pixels = distance_transform_edt(~mask)
    dist_meters = dist_pixels * px_size

    # Sample distances at centroid coordinates (set out-of-bounds to NaN)
    geoms = buildings.centroid
    rows, cols = rasterio.transform.rowcol(lu_meta["transform"], geoms.x, geoms.y)
    dist_values = np.full(len(geoms), np.nan)
    valid = (
        (rows >= 0) & (rows < lu_raster.shape[0]) &
        (cols >= 0) & (cols < lu_raster.shape[1])
    )
    dist_values[valid] = dist_meters[rows[valid], cols[valid]]

    return pd.Series(dist_values, index=buildings.index)


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
