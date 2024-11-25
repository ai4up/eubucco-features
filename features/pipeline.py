from momepy import longest_axis_length, elongation, convexity, orientation, corners, shared_walls

from log import setup_logger, LoggingContext
from util import load_buildings, load_streets
from features import building, buffer, street


def execute_feature_pipeline(city_path: str, log_file: str):
    logger = setup_logger(log_file=log_file)

    buildings = load_buildings(city_path)
    streets = load_streets(city_path)

    with LoggingContext(logger, feature_name='building'):
        buildings = _calculate_building_features(buildings)

    with LoggingContext(logger, feature_name='building_buffer'):
        buildings = _calculate_building_buffer_features(buildings)

    with LoggingContext(logger, feature_name='street'):
        buildings = _calculate_street_features(buildings, streets)


def _calculate_building_features(buildings):
        buildings['footprint_area'] = buildings.geometry.area
        buildings['perimeter'] = buildings.geometry.length
        buildings['normalized_perimeter_index'] = building.calculate_norm_perimeter(buildings)
        buildings['area_perimeter_ratio'] = buildings['footprint_area'] / buildings['perimeter']
        buildings['phi'] = building.calculate_phi(buildings)
        buildings['longestAxisLength'] = longest_axis_length(buildings)
        buildings['elongation'] = elongation(buildings)
        buildings['convexity'] = convexity(buildings)
        buildings['orientation'] = orientation(buildings)
        buildings['corners'] = corners(buildings)
        buildings['shared_wall_length'] = shared_walls(buildings)
        buildings['touches'] = building.calculate_touches(buildings)

        return buildings


def _calculate_building_buffer_features(buildings):
    buffer_fts = {
        'avg_footprint_area': ('footprint_area', 'mean'),
        'std_footprint_area': ('footprint_area', 'std'),
        'avg_elongation': ('elongation', 'mean'),
        'std_elongation': ('elongation', 'std'),
        'avg_convexity': ('convexity', 'mean'),
        'std_convexity': ('convexity', 'std'),
        'avg_orientation': ('orientation', 'mean'),
        'std_orientation': ('orientation', 'std'),
        'total_footprint_area': ('footprint_area', 'sum'),
        'n_buildings': ('footprint_area', 'count'),
    }
    buildings = buffer.calculate_h3_buffer_features(buildings, buffer_fts, 10, [1, 4])  # corresponds to a buffer of 0.1 and 0.9 km^2

    return buildings


def _calculate_street_features(buildings, streets):
    buildings[['size_of_closest_street', 'distance_to_closest_street']] = street.street_size_and_distance_to_closest_street(buildings, streets)

    return buildings


if __name__ == '__main__':
    city_path = 'test_data/Vaugneray'
    log_file = 'test_data/logs/features.log'
    execute_feature_pipeline(city_path, log_file)
