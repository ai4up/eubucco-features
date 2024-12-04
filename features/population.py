import geopandas as gpd
import pandas as pd

import util
from features import buffer


def count_local_population(buildings: gpd.GeoDataFrame, population_file: str) -> pd.Series:
    area = util.bbox(buildings, buffer=1000)
    pop = util.load_population(population_file, area, point_geom=False)
    pop = pop.to_crs(buildings.crs)

    bldg_centroids = buildings.centroid.to_frame()
    pop = gpd.sjoin(bldg_centroids, pop, how="left", predicate="within")["population"]

    return pop


def count_population_in_buffer(buildings: gpd.GeoDataFrame, population_file: str, h3_res: int) -> pd.Series:
    area = util.bbox(buildings, buffer=1000)
    pop = util.load_population(population_file, area, point_geom=True)

    h3_idx = f"h3_{h3_res}"
    buffer_fts = {"total_population": ("population", "sum")}
    hex_grid = buffer.aggregate_to_h3_grid(pop, buffer_fts, h3_res)

    buildings[h3_idx] = buffer.h3_index(buildings, h3_res)
    total_pop = buildings.merge(hex_grid, left_on=h3_idx, right_index=True, how="left")["population"]

    return total_pop
