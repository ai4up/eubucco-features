import math

import numpy as np
import geopandas as gpd

def calculate_phi(buildings):
    max_dist = buildings.geometry.map(lambda g: g.centroid.hausdorff_distance(g.exterior))
    circle_area = buildings.centroid.buffer(max_dist).area
    return buildings.area / circle_area


def calculate_touches(buildings):
    touching_pairs = gpd.sjoin(buildings, buildings, predicate='intersects')
    touches = touching_pairs.groupby('id_left').size() - 1
    return buildings['id'].map(touches).fillna(0).astype(int)


def calculate_norm_perimeter(buildings):
    return _circle_perimeter(buildings.area) / buildings.length


def _circle_perimeter(area):
    return 2 * np.sqrt(area * math.pi)
