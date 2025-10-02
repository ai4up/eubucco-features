from collections.abc import Iterable
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
import geopandas as gpd
import h3
import h3pandas  # noqa
import pandas as pd
from pyproj import Transformer
from shapely import Point


def aggregate_to_h3_grid(
    gdf: gpd.GeoDataFrame, operation: Dict[str, Tuple[str, Callable]], res: int
) -> pd.DataFrame:
    """
    Aggregates a GeoDataFrame to a hexagonal H3-indexed grid.

    Args:
        gdf: GeoDataFrame to be aggregated.
        operation:  A dictionary specifying aggregation operations for the buffer features. The keys specify
                    the new aggregated column names, and the values are tuples specifying the column name
                    and the aggregation function to use.
        res: Resolution of the hexagonal grid.

    Returns:
        Aggregated DataFrame.
    """
    if "h3_index" not in gdf.columns:
        gdf["h3_index"] = h3_index(gdf, res)

    hex_grid = gdf.groupby("h3_index").agg(**operation)

    return hex_grid


def calculate_h3_grid_shares(gdf: gpd.GeoDataFrame, col: str, res: int) -> gpd.GeoDataFrame:
    """
    Calculate the proportions of unique values in a column for each H3 hexagonal grid cell.

    Args:
        gdf: A GeoDataFrame containing the geometries and data.
        col: The column name to calculate proportions for.
        res: The resolution of the H3 hexagonal grid.

    Returns:
        A DataFrame with the unique value shares of the specified column within each H3 hexagonal grid cell.
    """
    if "h3_index" not in gdf.columns:
        gdf["h3_index"] = h3_index(gdf, res)

    hex_grid = gdf.groupby(["h3_index", col]).size() / gdf.groupby("h3_index").size()

    return hex_grid


def calculate_h3_buffer_features(
    gdf: gpd.GeoDataFrame, operation: Dict[str, Tuple[str, Callable]], res: int, k: Union[int, List[int]], grid_cells: pd.DataFrame = None
) -> gpd.GeoDataFrame:
    """
    Calculate buffer features for a GeoDataFrame based on H3 indexes.

    Args:
        gdf: A GeoDataFrame to be aggreated.
        operation:  A dictionary specifying aggregation operations for the buffer features. The keys are the names
                    of the features, and the values are tuples specifying the column name and the
                    aggregation function to use.
        res: H3 resolution level.
        k:  The number of hexagonal rings to include in the buffer. Provide a list to calculate features
            for multiple buffer sizes.
        grid_cells: Optional list of H3 indexes to calculate features for.

    Returns:
        A hexagonal grid with the calculated buffer features.
    """
    grid_values = aggregate_to_h3_grid(gdf, operation, res)
    if grid_cells is None:
        grid_cells = grid_values
    nbh_operation = _determine_neighborhood_agg_operation(operation)
    agg_grid = _calcuate_hex_rings_aggregate(grid_cells, grid_values, nbh_operation, res, k)

    return agg_grid


def calculate_h3_buffer_shares(
    gdf: gpd.GeoDataFrame, col: str, h3_res: int, k: Union[int, List[int]], grid_cells: pd.DataFrame = None
) -> gpd.GeoDataFrame:
    grid_shares = calculate_h3_grid_shares(gdf, col, h3_res)
    grid_shares = grid_shares.unstack(level=col, fill_value=0)
    if grid_cells is None:
        grid_cells = grid_shares
    agg_grid_shares = _calcuate_hex_rings_aggregate(grid_cells, grid_shares, "mean", h3_res, k)

    return agg_grid_shares


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
    centroids = gdf.centroid.to_crs("EPSG:4326")
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


def ft_suffix(res: int, k: int = 0) -> str:
    area = _calculate_buffer_area(res, k)
    return f"within_buffer_{area:.2f}km2"


def _calcuate_hex_rings_aggregate(
    grid_cells: pd.DataFrame, grid_values: pd.DataFrame, operation: Union[str, List, Dict, Callable], res: int, k: Union[int, List[int]]
) -> pd.DataFrame:
    aggregates = []
    hex_rings = _ensure_iterable(k)

    # Calculate aggregate for each hex ring size / buffer size
    for j in hex_rings:
        ring_aggregate = _calcuate_hex_ring_aggregate(grid_cells, grid_values, operation, j)
        ring_aggregate = ring_aggregate.add_suffix("_" + ft_suffix(res, j))
        aggregates.append(ring_aggregate)

    return pd.concat(aggregates, axis=1)


def _calcuate_hex_ring_aggregate(
    grid_cells: pd.DataFrame, grid_values: pd.DataFrame, operation: Union[str, List, Dict, Callable], k: int
) -> pd.DataFrame:
    # Add column with neighboring hexagons
    neighbors = grid_cells.h3.k_ring(k=k)["h3_k_ring"]

    # Add self to the neighbor list
    neighbors[:] = [[i] + n for i, n in zip(neighbors.index, neighbors)]

    # Perform aggregate operation (e.g. mean) across the hexagons in the neighborhood
    agg = neighbors.apply(lambda x: grid_values.reindex(x).agg(operation))

    return agg


def _h3_to_geo(h: str, crs: str = "EPSG:4326") -> Point:
    lat, lng = h3.h3_to_geo(h)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lng, lat)

    return Point(x, y)


def _ensure_iterable(var):
    if isinstance(var, Iterable) and not isinstance(var, (str, bytes)):
        return var

    return [var]


def _determine_neighborhood_agg_operation(op):
    # The two-step aggregation approach does not allow for exact calculation of some aggregates (e.g. std).
    # We need approximate these using the mean, while others can be calculated
    # exactly with another operation (e.g. count -> sum).
    operation_mapping = {
        "std": "mean",
        "nunique": "mean",
        "count": "sum",
        "sum": "sum",
        "mean": "mean",
        "max": "max",
        "min": "min",
    }
    try:
        for k, v in op.items():
            op[k] = operation_mapping[v[1]]

    except KeyError as e:
        raise Exception("Specific aggregation operation not (yet) supported.") from e

    return op


def _calculate_buffer_area(res, k):
    # areas in km2, from https://h3geo.org/docs/core-library/restable/#average-area-in-km2
    hex_areas = [
        4.3574e06,
        6.0978e05,
        8.6801e04,
        1.2393e04,
        1.7703e03,
        2.5290e02,
        3.6129e01,
        5.1612e00,
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
    n_hex_cells = 3 * (k**2) - 3 * k + 1
    buffer_area = n_hex_cells * hex_areas[res]

    return buffer_area
