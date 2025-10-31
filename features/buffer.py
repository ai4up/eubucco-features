from collections.abc import Iterable
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
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


def calculate_h3_grid_shares(gdf: gpd.GeoDataFrame, col: str, res: int, dropna: bool = False) -> gpd.GeoDataFrame:
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

    if dropna:
        gdf = gdf.dropna(subset=[col])

    hex_grid = gdf.groupby(["h3_index", col], observed=False).size()

    return hex_grid


def add_h3_buffer_mean_excluding_self(
    gdf: gpd.GeoDataFrame, cols: Dict[str, str], res: int, k: Union[int, List[int]], grid_cells: pd.DataFrame = None
) -> gpd.GeoDataFrame:
    """
        Calculate leaf-one-out average in buffer for a GeoDataFrame based on H3 indexes.

    Args:
        gdf: A GeoDataFrame to be aggreated.
        cols:  A dictionary specifying the columns to calculate leave-one-out means for. The keys are the names
               of the new features, and the values are the column names to calculate the means for.
        res: H3 resolution level.
        k:  The number of hexagonal rings to include in the buffer. Provide a list to calculate features
            for multiple buffer sizes.
        grid_cells: Optional list of H3 indexes to calculate features for.

    Returns:
        A hexagonal grid with the calculated buffer features.
    """
    operations = {f"_{op}_{col}": (col, op) for col in cols.values() for op in ["sum", "count"]}
    grid = calculate_h3_buffer_features(gdf, operations, res, k, grid_cells)
    gdf = gdf.merge(grid, left_on="h3_index", right_index=True, how="left")

    for col_mean, col in cols.items():
        for j in _ensure_iterable(k):
            suffix = ft_suffix(res, j)
            sum_col = f"_sum_{col}_{suffix}"
            count_col = f"_count_{col}_{suffix}"
            loo_mean_col = f"{col_mean}_{suffix}"

            na_mask = gdf[col].isna()
            gdf.loc[na_mask, loo_mean_col] = gdf[sum_col] / gdf[count_col]
            gdf.loc[~na_mask, loo_mean_col] = (gdf[sum_col] - gdf[col]) / (gdf[count_col] - 1)
            gdf.loc[gdf[count_col] <= 1, loo_mean_col] = np.nan
            gdf = gdf.drop(columns=[sum_col, count_col])

    return gdf


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
    agg_grid = _calculate_hex_rings_aggregate(grid_cells, grid_values, nbh_operation, res, k)

    return agg_grid


def calculate_h3_buffer_shares(
    gdf: gpd.GeoDataFrame, col: str, h3_res: int, k: Union[int, List[int]], grid_cells: pd.DataFrame = None, dropna: bool = False, n_min: int = 1, exclude_self: bool = False
) -> gpd.GeoDataFrame:
    grid_counts = calculate_h3_grid_shares(gdf, col, h3_res, dropna)
    grid_counts = grid_counts.unstack(level=col, fill_value=0)
    if grid_cells is None:
        grid_cells = grid_counts
    agg_grid = _calculate_hex_rings_aggregate(grid_cells, grid_counts, "sum", h3_res, k)
    exploded_grid = gdf[["h3_index", col]].merge(agg_grid, left_on="h3_index", right_index=True, how="left")

    ft_suffixes = [ft_suffix(h3_res, j) for j in _ensure_iterable(k)]
    for suffix in ft_suffixes:
        if exclude_self:
            for t, idx in exploded_grid.groupby(col, dropna=True, observed=True).groups.items():
                exploded_grid.loc[idx, f'{t}_{suffix}'] -= 1

        counts = exploded_grid.filter(like=suffix)
        totals = counts.sum(axis=1)
        shares = counts.div(totals, axis=0)
        shares[totals < n_min] = np.nan
        exploded_grid[counts.columns] = shares

    return exploded_grid.drop(columns=["h3_index", col])


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


def ft_suffix(res: int, k: int = 0) -> str:
    area = _calculate_buffer_area(res, k)
    return f"within_buffer_{area:.2f}km2"


def _calculate_hex_rings_aggregate(
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
