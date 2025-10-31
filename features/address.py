import geopandas as gpd
import pandas as pd

from util import count_dwithin, distance_nearest


def load_addresses(addresses_path: str, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bbox = buildings.total_bounds
    addresses = gpd.read_parquet(addresses_path, bbox=bbox)

    return addresses


def building_address_count(buildings: gpd.GeoDataFrame, addresses: gpd.GeoDataFrame, tolerance: float = 10) -> pd.Series:
    addr_counts = count_dwithin(buildings, addresses, distance=tolerance)

    return addr_counts


def building_address_unit_count(buildings: gpd.GeoDataFrame, addresses: gpd.GeoDataFrame, tolerance: float = 10) -> pd.Series:
    address_units = addresses[addresses["number"].astype(str).str.contains(r"[A-Za-z]").fillna(False)]
    addr_counts = count_dwithin(buildings, address_units, distance=tolerance)

    return addr_counts


def distance_to_closest_address(buildings: gpd.GeoDataFrame, addresses: gpd.GeoDataFrame) -> gpd.GeoSeries:
    if "address_count" in buildings.columns:
        dis = pd.Series(index=buildings.index)
        mask = buildings["address_count"] == 0
        dis[~mask] = 0
        dis[mask] = distance_nearest(buildings[mask].centroid, addresses, max_distance=100).fillna(100)
    else:
        dis = distance_nearest(buildings.centroid, addresses, max_distance=100).fillna(100)

    return dis
