import pandas as pd
import geopandas as gpd
from shapely import wkt
from pathlib import Path
from typing import Optional

CRS_UNI = 'EPSG:3035'
geometry_col = 'geometry'


def load_csv(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=df[geometry_col].apply(wkt.loads),
        crs=CRS_UNI
    )
    return gdf


def load_buildings(city_path: Path) -> gpd.GeoDataFrame:
    city_name = city_path.split("/")[-1]
    buildings = load_csv(Path(f"{city_path}/{city_name}_geom.csv"))
    buffer_ = load_csv(Path(f"{city_path}/{city_name}_buffer.csv"))
    buildings = gpd.pd.concat([buildings, buffer_], ignore_index=True)
    return buildings
