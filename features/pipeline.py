from momepy import longest_axis_length, elongation, convexity, equivalent_rectangular_index, orientation, corners, shared_walls
import numpy as np
import geopandas as gpd

from log import setup_logger, LoggingContext
from util import load_buildings, load_osm_buildings, load_streets, load_pois, store_features
from features import building, buffer, street, poi, osm, landuse

H3_RES = 10
H3_BUFFER_SIZES = [1, 4] # corresponds to a buffer of 0.1 and 0.9 km^2

def execute_feature_pipeline(city_path: str, log_file: str, lu_path: str):
    logger = setup_logger(log_file=log_file)

    buildings = load_buildings(city_path)
    buildings['h3_index'] = buffer.h3_index(buildings, H3_RES)

    with LoggingContext(logger, feature_name='building'):
        buildings = _calculate_building_features(buildings)

    with LoggingContext(logger, feature_name='street'):
        buildings = _calculate_street_features(buildings, city_path)

    with LoggingContext(logger, feature_name='poi'):
        buildings = _calculate_poi_features(buildings, city_path)

    with LoggingContext(logger, feature_name='osm_buildings'):
        buildings = _calculate_osm_buildings_features(buildings, city_path)

    with LoggingContext(logger, feature_name='landuse'):
        buildings = _calculate_landuse_features(buildings, lu_path)

    with LoggingContext(logger, feature_name='buffer'):
        buildings = _calculate_buffer_features(buildings)

    store_features(buildings, city_path)


def _calculate_building_features(buildings: gpd.GeoDataFrame):
        buildings['footprint_area'] = buildings.area
        buildings['perimeter'] = buildings.length
        buildings['normalized_perimeter_index'] = building.calculate_norm_perimeter(buildings)
        buildings['area_perimeter_ratio'] = buildings['footprint_area'] / buildings['perimeter']
        buildings['phi'] = building.calculate_phi(buildings)
        buildings['longestAxisLength'] = longest_axis_length(buildings)
        buildings['elongation'] = elongation(buildings)
        buildings['convexity'] = convexity(buildings)
        buildings['rectangularity'] = equivalent_rectangular_index(buildings)
        buildings['orientation'] = orientation(buildings)
        buildings['corners'] = corners(buildings)
        buildings['shared_wall_length'] = shared_walls(buildings)
        buildings['touches'] = building.calculate_touches(buildings)

        return buildings


def _calculate_buffer_features(buildings: gpd.GeoDataFrame):
    buffer_fts = {
        'avg_footprint_area': ('footprint_area', 'mean'),
        'std_footprint_area': ('footprint_area', 'std'),
        'max_footprint_area': ('footprint_area', 'max'),
        'avg_elongation': ('elongation', 'mean'),
        'std_elongation': ('elongation', 'std'),
        'max_elongation': ('elongation', 'max'),
        'avg_convexity': ('convexity', 'mean'),
        'std_convexity': ('convexity', 'std'),
        'max_convexity': ('convexity', 'max'),
        'avg_orientation': ('orientation', 'mean'),
        'std_orientation': ('orientation', 'std'),
        'max_orientation': ('orientation', 'max'),
        'avg_size_of_closest_street': ('size_of_closest_street', 'mean'),
        'std_size_of_closest_street': ('size_of_closest_street', 'std'),
        'max_size_of_closest_street': ('size_of_closest_street', 'max'),
        'avg_distance_to_closest_street': ('distance_to_closest_street', 'mean'),
        'std_distance_to_closest_street': ('distance_to_closest_street', 'std'),
        'max_distance_to_closest_street': ('distance_to_closest_street', 'max'),
        'total_footprint_area': ('footprint_area', 'sum'),
        'n_buildings': ('footprint_area', 'count'),
    }
    hex_grid = buffer.calculate_h3_buffer_features(buildings, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = buildings.merge(hex_grid, left_on='h3_index', right_index=True, how='left')

    return buildings


def _calculate_street_features(buildings: gpd.GeoDataFrame, city_path: str):
    streets = load_streets(city_path)

    buildings[['size_of_closest_street', 'distance_to_closest_street']] = street.street_size_and_distance_to_closest_street(buildings, streets)

    return buildings


def _calculate_poi_features(buildings: gpd.GeoDataFrame, city_path: str):
    pois = load_pois(city_path)

    buildings['distance_to_closest_poi'] = poi.distance_to_closest_poi(buildings, pois)

    buffer_fts = {'n_pois': ('amenity', 'count')}
    hex_grid = buffer.calculate_h3_buffer_features(pois, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = buildings.merge(hex_grid, left_on='h3_index', right_index=True, how='left')

    hex_grid_large_buffer = hex_grid[hex_grid.columns[-1]]
    buildings['distance_to_center'] = buffer.distance_to_h3_grid_max(buildings, hex_grid_large_buffer)

    return buildings


def _calculate_landuse_features(buildings: gpd.GeoDataFrame, lu_path: str):
    buildings['lu_distance_to_industry'] = landuse.distance_to_landuse(buildings, 'industrial', lu_path)
    buildings['lu_distance_to_agriculture'] = landuse.distance_to_landuse(buildings, 'agricultural', lu_path)

    return buildings


def _calculate_osm_buildings_features(buildings: gpd.GeoDataFrame, city_path: str):
    osm_buildings = load_osm_buildings(city_path)

    buildings = osm.closest_building_attributes(buildings, osm_buildings, {'type': 'osm_closest_building_type', 'height': 'osm_closest_building_height'})

    buildings['osm_distance_to_industry'] = osm.distance_to_some_building_type(buildings, osm_buildings, 'industrial')
    buildings['osm_distance_to_commercial'] = osm.distance_to_some_building_type(buildings, osm_buildings, 'commercial')
    buildings['osm_distance_to_agriculture'] = osm.distance_to_some_building_type(buildings, osm_buildings, 'agricultural')
    buildings['osm_distance_to_education'] = osm.distance_to_some_building_type(buildings, osm_buildings, 'education')

    buildings['osm_distance_to_medium_rise'] = osm.distance_to_some_building_height(buildings, osm_buildings, [15, 30])
    buildings['osm_distance_to_high_rise'] = osm.distance_to_some_building_height(buildings, osm_buildings, [30, np.inf])

    hex_grid_type_shares = osm.building_type_share_buffer(osm_buildings, H3_RES, H3_BUFFER_SIZES)
    hex_grid_type_shares = hex_grid_type_shares.add_prefix('osm_type_share_')
    buildings = buildings.merge(hex_grid_type_shares, left_on='h3_index', right_index=True, how='left')

    buffer_fts = {
        'osm_avg_height': ('height', 'mean'),
        'osm_std_height': ('height', 'std'),
        'osm_max_height': ('height', 'max'),
        'osm_type_variety': ('type', 'nunique'),
    }
    hex_grid = buffer.calculate_h3_buffer_features(osm_buildings, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = buildings.merge(hex_grid, left_on='h3_index', right_index=True, how='left')

    return buildings


if __name__ == '__main__':
    city_path = 'test_data/Toulouse'
    log_file = 'test_data/logs/features.log'
    corine_lu_path = 'test_data/U2018_CLC2018_V2020_20u1.gpkg'

    execute_feature_pipeline(city_path, log_file, corine_lu_path)
