from collections.abc import Iterable
from typing import Callable, Dict, List, Tuple, Union

import h3pandas
import pandas as pd
import h3
import geopandas as gpd


def calculate_h3_buffer_features(buildings: gpd.GeoDataFrame, operation: Dict[str, Tuple[str, Callable]], res: int, k: Union[int, List[int]]) -> gpd.GeoDataFrame:
    """
    Calculate buffer features for a GeoDataFrame based on H3 indexes.

    Parameters:
    - buildings (GeoDataFrame): A GeoDataFrame containing buildings.
    - operation (dict): A dictionary specifying aggregation operations for the buffer features. The keys are the names of the features, and the values are tuples specifying the column name and the aggregation function to use.
    - res (int): H3 resolution level.
    - k (int or List[int]): The number of hexagonal rings to include in the buffer. Can be a single value or a list of values.

    Returns:
    - gdf (GeoDataFrame): GeoDataFrame with the calculated buffer features added.

    Example usage:
    ```
    gdf = calculate_h3_buffer_features(gdf, operation, res, k)
    ```
    """
    buildings['h3_index'] = _h3_index(buildings, res)
    hex_grid = buildings.groupby('h3_index').agg(**operation)
    nbh_operation = {ft_name: v[1] for ft_name, v in operation.items()}
    hex_grid = pd.concat([
        _calcuate_hex_ring_aggregate(
            hex_grid, j, nbh_operation
        ).add_suffix(
            f'_within_{_calculate_buffer_area(res, j):.2f}_buffer'
        )
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
    neighbors = gdf.h3.k_ring(k=k)['h3_k_ring']

    # Add self to the neighbor list
    neighbors[:] = [[i] + n for i, n in zip(neighbors.index, neighbors)]

    # Perform aggregate operation (e.g. mean) across the hexagons in the neighborhood
    agg = neighbors.apply(lambda x: gdf.reindex(x).agg(operation))

    return agg


def _ensure_iterable(var):
    if isinstance(var, Iterable) and not isinstance(var, (str, bytes)):
        return var

    return [var]


def _calculate_buffer_area(res, k):
    # areas in km2, from https://h3geo.org/docs/core-library/restable/#average-area-in-km2
    hex_areas = [
        4.3574e+06,
        6.0978e+05,
        8.6801e+04,
        1.2393e+04,
        1.7703e+03,
        2.5290e+02,
        3.6129e+01,
        5.1612e+00,
        7.3732e-01,
        1.0533e-01,
        1.5047e-02,
        2.1496e-03,
        3.0709e-04,
        4.3870e-05,
        6.2671e-06,
        8.9531e-07,
    ]
    k = k + 1
    n_hex_cells = 3 * (k ** 2) - 3 * k + 1
    buffer_area = n_hex_cells * hex_areas[res]

    return buffer_area
