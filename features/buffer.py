from collections.abc import Iterable
from typing import Callable, Dict, List, Tuple, Union

import h3pandas
import pandas as pd
import h3
import geopandas as gpd


def calculate_h3_buffer_features(buildings: gpd.GeoDataFrame, buffer_fts: Dict[str, Tuple[str, Callable]], res: int, k: Union[int, List[int]]) -> gpd.GeoDataFrame:
    """
    Calculate buffer features for a GeoDataFrame based on H3 indexes.

    Parameters:
    - buildings (GeoDataFrame): A GeoDataFrame containing buildings.
    - buffer_fts (dict): A dictionary specifying the buffer features to calculate. The keys are the names of the features, and the values are tuples specifying the column name and the aggregation function to use.
    - res (int): H3 resolution level.
    - k (int or List[int]): The number of hexagonal rings to include in the buffer. Can be a single value or a list of values.

    Returns:
    - gdf (GeoDataFrame): GeoDataFrame with the calculated buffer features added.

    Example usage:
    ```
    gdf = calculate_h3_buffer_features(gdf, buffer_fts, res, k)
    ```
    """
    buildings['h3_index'] = _h3_index(buildings, res)
    hex_grid = buildings.groupby('h3_index').agg(**buffer_fts)
    nghb_agg = {ft_name: v[1] for ft_name, v in buffer_fts.items()}
    hex_grid = pd.concat([
        _calcuate_hex_ring_aggregate(hex_grid, k=j, operation=nghb_agg).add_suffix(f'_within_h3_res{res}_buffer_k{j}')
        for j in _ensure_iterable(k)
    ], axis=1)
    buildings = buildings.merge(hex_grid, left_on='h3_index', right_index=True, how='left')

    return buildings


def _h3_index(gdf: Union[gpd.GeoSeries, gpd.GeoDataFrame], res: int) -> List[str]:
    # H3 operations require a lat/lon point geometry
    centroids = gdf.centroid.to_crs('EPSG:4326')
    lngs = centroids.x
    lats = centroids.y
    h3_idx = [h3.geo_to_h3(lat, lng, res) for lat, lng in zip(lats, lngs)]

    return h3_idx


def _calcuate_hex_ring_aggregate(gdf: gpd.GeoDataFrame, k: int, operation: Union[str, List, Dict, Callable[[float], float]]) -> gpd.GeoDataFrame:
    # Add column with neighboring hexagons
    neighbors = gdf.h3.hex_ring(k=k)['h3_hex_ring']

    # Add self to the neighbor list
    neighbors[:] = [[i] + n for i, n in zip(neighbors.index, neighbors)]

    # Perform aggregate operation (e.g. mean) across the hexagons in the neighborhood
    agg = neighbors.apply(lambda x: gdf.reindex(x).agg(operation))

    return agg

def _ensure_iterable(var):
    if isinstance(var, Iterable) and not isinstance(var, (str, bytes)):
        return var

    return [var]
