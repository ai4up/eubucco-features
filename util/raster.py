from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.mask
import rasterio.transform
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box


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

        # Mask out any invalid data (non-urban regions)
        data = np.ma.masked_less(data, 0)

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


def read_values(filepath: str, points: gpd.GeoSeries) -> pd.Series:
    with rasterio.open(filepath) as src:
        if points.crs != src.crs:
            points = points.to_crs(src.crs)

        # Extract raster values for (x, y) pairs
        coords = [(pt.x, pt.y) for pt in points]
        values = list(src.sample(coords))
        values = np.array(values).squeeze()

        # Mask missings
        values = np.where(values == src.nodata, np.nan, values)

    return pd.Series(values, index=points.index)


def distance_nearest_cell(gdf: gpd.GeoDataFrame, raster_data: np.ndarray, meta: dict, mask: np.ndarray) -> pd.Series:
    if not np.any(mask):
        return pd.Series(np.nan, index=gdf.index)

    # Compute distance transform (in meters)
    px_size = meta["transform"].a
    dist_pixels = distance_transform_edt(~mask)
    dist_meters = dist_pixels * px_size

    # Sample distances at centroid coordinates (set out-of-bounds to NaN)
    geoms = gdf.centroid
    rows, cols = rasterio.transform.rowcol(meta["transform"], geoms.x, geoms.y)
    dist_values = np.full(len(geoms), np.nan)
    valid = (
        (rows >= 0) & (rows < raster_data.shape[0]) &
        (cols >= 0) & (cols < raster_data.shape[1])
    )
    dist_values[valid] = dist_meters[rows[valid], cols[valid]]

    return pd.Series(dist_values, index=gdf.index)
