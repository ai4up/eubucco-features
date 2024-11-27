from collections.abc import Iterable
from typing import Callable, Dict, List, Tuple, Union

from shapely import Point
from pyproj import Transformer
import h3pandas
import pandas as pd
import h3
import geopandas as gpd


def aggregate_to_h3_grid(gdf: gpd.GeoDataFrame, operation: Dict[str, Tuple[str, Callable]], res: int) -> gpd.GeoDataFrame:
    """
    Aggregates a GeoDataFrame to a hexagonal H3-indexed grid.

    Args:
        gdf: GeoDataFrame to be aggregated.
        operation: A dictionary specifying aggregation operations for the buffer features. The keys specify the new aggregated column names, and the values are tuples specifying the column name and the aggregation function to use.
        res: Resolution of the hexagonal grid.

    Returns:
        Aggregated GeoDataFrame.
    """
    if 'h3_index' not in gdf.columns:
        gdf['h3_index'] = h3_index(gdf, res)

    hex_grid = gdf.groupby('h3_index').agg(**operation)

    return hex_grid


def calculate_h3_grid_shares(gdf: gpd.GeoDataFrame, col: str, res: int) -> gpd.GeoDataFrame:
    """
    Calculate the proportions of unique values in a column for each H3 hexagonal grid cell.

    Args:
        gdf: A GeoDataFrame containing the geometries and data.
        col: The column name to calculate proportions for.
        res: The resolution of the H3 hexagonal grid.

    Returns:
        A GeoDataFrame with the unique value shares of the specified column within each H3 hexagonal grid cell.
    """
    if 'h3_index' not in gdf.columns:
        gdf['h3_index'] = h3_index(gdf, res)

    hex_grid = gdf.groupby(['h3_index', col]).size() / gdf.groupby('h3_index').size()

    return hex_grid


def calculate_h3_buffer_features(gdf: gpd.GeoDataFrame, operation: Dict[str, Tuple[str, Callable]], res: int, k: Union[int, List[int]]) -> gpd.GeoDataFrame:
    """
    Calculate buffer features for a GeoDataFrame based on H3 indexes.

    Args:
        gdf: A GeoDataFrame to be aggreated.
        operation: A dictionary specifying aggregation operations for the buffer features. The keys are the names of the features, and the values are tuples specifying the column name and the aggregation function to use.
        res: H3 resolution level.
        k: The number of hexagonal rings to include in the buffer. Provide a list to calculate features for multiple buffer sizes.

    Returns:
        A hexagonal grid with the calculated buffer features.
    """
    hex_grid = aggregate_to_h3_grid(gdf, operation, res)
    nbh_operation = {ft_name: v[1] for ft_name, v in operation.items()}
    hex_grid = _calcuate_hex_rings_aggregate(hex_grid, nbh_operation, res, k)

    return hex_grid


def h3_index(gdf: Union[gpd.GeoSeries, gpd.GeoDataFrame], res: int) -> List[str]:
    """
    Generate H3 indexes for the geometries in a GeoDataFrame or GeoSeries.

    Args:
        gdf: A GeoSeries or GeoDataFrame.
        res: The resolution of the H3 index.

    Returns:
        A list of H3 indexes corresponding to the input geometries.
    """
    # H3 operations require a lat/lon point geometry
    centroids = gdf.centroid.to_crs('EPSG:4326')
    lngs = centroids.x
    lats = centroids.y
    h3_idx = [h3.geo_to_h3(lat, lng, res) for lat, lng in zip(lats, lngs)]

    return h3_idx


def distance_to_h3_grid_max(gdf: gpd.GeoDataFrame, s: pd.Series):
    """
    Calculate the distance from each geometry in a GeoDataFrame to the geometry
    corresponding to the maximum value in a given H3-indexed Series.

    Args:
        gdf: A GeoDataFrame.
        s: A Series with an H3 index.

    Returns:
        A Series containing the distances to the location with the maximum value.
    """
    h3_peak = s.idxmax()
    peak = _h3_to_geo(h3_peak, gdf.crs)
    dis = gdf.distance(peak)

    return dis


def _calcuate_hex_rings_aggregate(hex_grid: gpd.GeoDataFrame, operation: Union[str, List, Dict, Callable], res: int, k: Union[int, List[int]]) -> gpd.GeoDataFrame:
    aggregates = []
    hex_rings = _ensure_iterable(k)

    # Calculate aggregate for each hex ring size / buffer size
    for j in hex_rings:
        buffer_area = _calculate_buffer_area(res, j)
        ring_aggregate = _calcuate_hex_ring_aggregate(hex_grid, j, operation)
        ring_aggregate = ring_aggregate.add_suffix(f'_within_{buffer_area:.2f}_buffer')
        aggregates.append(ring_aggregate)

    return pd.concat(aggregates, axis=1)


def _calcuate_hex_ring_aggregate(gdf: gpd.GeoDataFrame, operation: Union[str, List, Dict, Callable], k: int) -> gpd.GeoDataFrame:
    # Add column with neighboring hexagons
    neighbors = gdf.h3.k_ring(k=k)['h3_k_ring']

    # Add self to the neighbor list
    neighbors[:] = [[i] + n for i, n in zip(neighbors.index, neighbors)]

    # Perform aggregate operation (e.g. mean) across the hexagons in the neighborhood
    agg = neighbors.apply(lambda x: gdf.reindex(x).agg(operation))

    return agg


def _h3_to_geo(h: str, crs: str = 'EPSG:4326') -> Point:
    lat, lng = h3.h3_to_geo(h)
    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    x, y = transformer.transform(lng, lat)

    return Point(x, y)


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
