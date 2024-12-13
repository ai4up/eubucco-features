import os
import uuid
from pathlib import Path
from typing import Callable, Union

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon

import util

CRS_UNI = "EPSG:3035"
geometry_col = "geometry"


def load_csv(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col].apply(wkt.loads), crs=CRS_UNI)
    return gdf


def load_buildings(city_path: str) -> gpd.GeoDataFrame:
    city_name = city_path.split("/")[-1]
    buildings = load_csv(Path(f"{city_path}/{city_name}_geom.csv"))
    buffer_ = load_csv(Path(f"{city_path}/{city_name}_buffer.csv"))
    buffer_["id"] = [uuid.uuid4() for _ in range(len(buffer_))]  # temporary fix for missing ids
    buildings = gpd.pd.concat([buildings, buffer_], ignore_index=True)
    return buildings


def load_streets(city_path: str) -> gpd.GeoDataFrame:
    city_name = city_path.split("/")[-1]
    streets = load_csv(Path(f"{city_path}/{city_name}_streets_raw.csv"))
    return streets


def load_pois(city_path: str) -> gpd.GeoDataFrame:
    city_name = city_path.split("/")[-1]
    pois = gpd.read_file(Path(f"{city_path}/{city_name}_pois.gpkg"))
    return pois


def load_osm_buildings(city_path: str) -> gpd.GeoDataFrame:
    city_name = city_path.split("/")[-1]
    buildings = gpd.read_file(Path(f"{city_path}/{city_name}_osm_bldgs.gpkg"))
    return buildings


def load_population(population_file: str, area: gpd.GeoSeries, point_geom: bool) -> gpd.GeoDataFrame:
    population_raster, city_meta = util.read_area(population_file, area)
    population = util.raster_to_gdf(population_raster[0], city_meta, point=point_geom)
    population = population.rename(columns={"values": "population"})

    return population


def load_elevation(elevation_file: str, area: gpd.GeoSeries, point_geom: bool) -> gpd.GeoDataFrame:
    elevation_raster, city_meta = util.read_area(elevation_file, area)
    elevation = util.raster_to_gdf(elevation_raster[0], city_meta, point=point_geom)
    elevation = elevation.rename(columns={"values": "elevation"})

    return elevation


def load_GHS_built_up(built_up_file: str, area: gpd.GeoSeries) -> gpd.GeoDataFrame:
    built_up_raster, city_meta = util.read_area(built_up_file, area)
    built_up = util.raster_to_gdf(built_up_raster[0], city_meta)
    built_up = built_up.rename(columns={"values": "class"})

    use_types = {
        "residential": [11, 12, 13, 14, 15],
        "non-residential": [21, 22, 23, 24, 25],
    }
    heights = {
        4.5: [12, 22],  # 3-6m
        10.5: [13, 23],  # 6-15m
        22.5: [14, 24],  # 15-30m
    }
    greeness_NDVI = {
        1: 0.15,  # low vegetation surfaces NDVI <= 0.3
        2: 0.4,  # medium vegetation surfaces 0.3 < NDVI <=0.5
        3: 0.75,  # high vegetation surfaces NDVI > 0.5
    }

    def reverse(d):
        return {value: key for key, values in d.items() for value in values}

    built_up["height"] = built_up["class"].map(reverse(heights))
    built_up["high_rise"] = built_up["class"].isin([15, 25])
    built_up["use_type"] = built_up["class"].map(reverse(use_types))
    built_up["NDVI"] = built_up["class"].map(greeness_NDVI)

    return built_up


def store_features(buildings: gpd.GeoDataFrame, city_path: str):
    city_name = city_path.split("/")[-1]
    buildings.to_file(Path(f"{city_path}/{city_name}_features.gpkg"), driver="GPKG")


def nuts_geometries(nuts_path: str, crs: str, buffer: int = 0) -> (str, Union[Polygon, MultiPolygon]):
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
