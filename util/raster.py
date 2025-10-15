from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.mask
import rasterio.transform
from rasterio.transform import rowcol
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from scipy.ndimage import maximum_filter, uniform_filter


def raster_to_gdf(
    raster_data: Union[np.ndarray, np.ma.MaskedArray], meta: dict, point: bool = True
) -> gpd.GeoDataFrame:
    rows, cols = raster_data.shape
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))

    if point:
        xs, ys = rasterio.transform.xy(meta["transform"], ys, xs, offset="center")
        xs = np.array(xs).flatten()
        ys = np.array(ys).flatten()
        geom = gpd.points_from_xy(xs, ys)
    else:
        x_min, y_min = rasterio.transform.xy(meta["transform"], ys, xs, offset="ul")
        x_max, y_max = rasterio.transform.xy(meta["transform"], ys, xs, offset="lr")
        x_min = np.array(x_min).flatten()
        x_max = np.array(x_max).flatten()
        y_min = np.array(y_min).flatten()
        y_max = np.array(y_max).flatten()
        geom = [box(*box_coords) for box_coords in list(zip(x_min, y_min, x_max, y_max))]

    values = raster_data.flatten()
    values = pd.Series(values, name="values")

    gdf = gpd.GeoDataFrame(values, geometry=geom, crs=meta["crs"])

    return gdf


def read_area(filepath: str, geometries: gpd.GeoSeries):
    with rasterio.open(filepath) as src:
        # Convert to CRS of TIF file
        geometries = geometries.to_crs(src.crs).values

        # Read rastered data only in specific areas
        data, out_transform = rasterio.mask.mask(src, geometries, crop=True)

        # Convert to float array for NaN support
        data = data.astype(float)

        # Replace nodata values with np.nan
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

        # Update meta data
        city_meta = src.meta.copy()
        city_meta.update({"height": data.shape[1], "width": data.shape[2], "transform": out_transform})

    return data, city_meta


def read_value(filepath: str, lon: float, lat: float, approx: bool = False):
    with rasterio.open(filepath, crs="EPSG:4326") as src:
        coord = src.index(lon, lat)
        value = src.read(1)[coord]  # first band

        if approx and np.isnan(value):
            print(f"Value at lat {lat:.3f}, lon {lon:.3f} is NaN. Calculating average over 3x3 grid around location.")
            x_min = coord[0] - 1
            x_max = coord[0] + 2
            y_min = coord[1] - 1
            y_max = coord[1] + 2
            values = src.read(1)[x_min:x_max, y_min:y_max]
            value = np.nanmean(values)

    return value


def read_few_values(filepath: str, points: gpd.GeoSeries) -> pd.Series:
    with rasterio.open(filepath) as src:

        if points.geometry.type != "Point":
            points = points.centroid

        if points.crs != src.crs:
            points = points.to_crs(src.crs)

        coords = [(pt.x, pt.y) for pt in points]
        values = list(src.sample(coords))
        values = np.array(values).squeeze()

        if src.nodata is not None:
            values[values == src.nodata] = np.nan

    return pd.Series(values, index=points.index)


def read_values(points: gpd.GeoSeries, raster_data: np.ndarray, meta: dict) -> pd.Series:
    rows, cols = _geom_to_rowcol(points, meta["transform"], meta["crs"])
    values = raster_data[rows, cols]

    return pd.Series(values, index=points.index)


def read_values_pooled(points: gpd.GeoSeries, raster_data: np.ndarray, meta: dict, window_size: int) -> pd.Series:
    # Apply local max pooling
    data_filled = np.nan_to_num(raster_data, nan=-np.inf)
    pooled = maximum_filter(data_filled, size=window_size, mode="nearest")
    pooled[pooled == -np.inf] = np.nan

    # Convert (x, y) to raster indices
    rows, cols = _geom_to_rowcol(points, meta["transform"], meta["crs"])

    # Sample directly from the pooled array
    values = pooled[rows, cols]

    return pd.Series(values, index=points.index)


def distance_nearest_cell(points: gpd.GeoSeries, raster_data: np.ndarray, meta: dict, mask: np.ndarray) -> pd.Series:
    if not np.any(mask):
        return pd.Series(np.nan, index=points.index)

    # Compute distance transform (in meters)
    px_size = meta["transform"].a
    dist_pixels = distance_transform_edt(~mask)
    dist_meters = dist_pixels * px_size

    # Sample distances at coordinates
    rows, cols = _geom_to_rowcol(points, meta["transform"], meta["crs"])

    # Set out-of-bounds to NaN
    dist_values = np.full(len(points), np.nan)
    valid = (
        (rows >= 0) & (rows < raster_data.shape[0]) &
        (cols >= 0) & (cols < raster_data.shape[1])
    )
    dist_values[valid] = dist_meters[rows[valid], cols[valid]]

    return pd.Series(dist_values, index=points.index)


def area_mean(points: gpd.GeoSeries, raster_data: np.ndarray, meta: dict, buffer: int) -> pd.Series:
    px_buffer = _metric_buffer_to_px(buffer, meta["transform"])
    mean_values = _mean_pooling(raster_data, block_size=px_buffer)
    rows, cols = _geom_to_rowcol(points, meta["transform"], meta["crs"])
    point_values = mean_values[rows, cols]

    return pd.Series(point_values, index=points.index)


def map_values(arr: np.ndarray, mapping: dict, default_value=np.nan):
    """Remap raster classes to numeric values, keeping NaNs untouched."""
    out = np.full_like(arr, default_value, dtype=float)
    out[np.isnan(arr)] = np.nan
    for cls, new_val in mapping.items():
        out[arr == cls] = new_val

    return out


def _geom_to_rowcol(points: gpd.GeoSeries, transform: rasterio.Affine, crs: str) -> pd.Series:
    if (points.geometry.type != "Point").all():
        points = points.centroid

    if points.crs != crs:
        points = points.to_crs(crs)

    return rowcol(transform, points.x, points.y)


def _metric_buffer_to_px(buffer: int, transform: rasterio.Affine) -> int:
    px_size = transform.a
    px_radius = int(np.ceil(buffer / px_size))
    px_buffer = 2 * px_radius + 1

    return px_buffer


def _nanmean_pooling(raster_data: np.ndarray, size: int) -> np.ndarray:
    """Apply NaN-safe mean filter with rectangular kernel."""

    # Count of valid (non-NaN) pixels
    valid = (~np.isnan(raster_data)).astype(float)
    count = uniform_filter(valid, size=size, mode="nearest") * (size * size)

    # Sum of values in window (undoing uniform_filter's normalization)
    data_filled = np.nan_to_num(raster_data, nan=0.0)
    summed = uniform_filter(data_filled, size=size, mode="nearest") * (size * size)

    # Mean ignoring NaNs
    mean = np.divide(summed, count, where=count > 0.001)
    mean[count < 0.001] = np.nan  # keep fully-NaN windows as NaN

    return mean


def _mean_pooling(raster_data, block_size):
    if np.isnan(raster_data).any():
        pooled = _nanmean_pooling(raster_data, block_size)
    else:
        pooled = uniform_filter(np.float32(raster_data), size=block_size, mode="nearest")

    return pooled
