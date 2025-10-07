import os
from typing import Callable, Iterator, Tuple, Union

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


def load_buildings(buildings_dir: str, region_id: str) -> gpd.GeoDataFrame:
    bldgs_file = os.path.join(buildings_dir, f"{region_id}.parquet")
    buildings = gpd.read_parquet(bldgs_file)

    return buildings


def store_features(buildings: gpd.GeoDataFrame, out_dir: str, region_id: str):
    out_file = os.path.join(out_dir, f"{region_id}.parquet")
    buildings.to_parquet(out_file)


def nuts_geometries(nuts_path: str, crs: str, buffer: int = 0) -> Iterator[Tuple[str, Union[Polygon, MultiPolygon]]]:
    nuts = gpd.read_file(nuts_path)
    nuts = nuts.dissolve("NUTS_ID")

    if buffer:
        local_crs = nuts.estimate_utm_crs()
        nuts = nuts.to_crs(local_crs).buffer(buffer)

    nuts_geoms = nuts.to_crs(crs).geometry
    for nuts_id, nuts_geom in nuts_geoms.items():
        yield nuts_id, nuts_geom


def download_all_nuts(download_func: Callable, nuts_path: str, out_path: str, buffer: int = 0) -> None:
    for nuts_id, nuts_geom in nuts_geometries(nuts_path, crs="EPSG:4326", buffer=buffer):
        file_path = os.path.join(out_path, f"{nuts_id}.gpkg")

        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping download.")
            continue

        gdf = download_func(nuts_geom)

        if gdf is None:
            print(f"Download failed for NUTS region {nuts_id}.")
            continue

        gdf.to_file(file_path, driver="GPKG")


def load_gpkg(data_dir: str, region_id: str) -> gpd.GeoDataFrame:
    gdf_file = os.path.join(data_dir, f"{region_id}.gpkg")
    gdf = gpd.read_file(gdf_file)

    return gdf
