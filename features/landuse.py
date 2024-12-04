import geopandas as gpd
import pandas as pd

# CORINE landuse classes https://land.copernicus.eu/en/products/corine-land-cover
CORINE_LU_CLASS_COL = "Code_18"
CORINE_LU_CLASSES = {"agricultural": [211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244], "industrial": [121]}
CORINE_CRS = 3035


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
    bbox = buildings.to_crs(CORINE_CRS).total_bounds
    lu = gpd.read_file(landuse_path, bbox=tuple(bbox))

    lu = lu[lu[CORINE_LU_CLASS_COL].astype(int).isin(CORINE_LU_CLASSES[category])]
    lu = lu.union_all()
    dis = buildings.centroid.distance(lu)

    return dis
