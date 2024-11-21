from pathlib import Path

from log import setup_logger, LoggingContext
from util import load_buildings
from momepy import longest_axis_length, elongation, convexity, orientation, corners


def execute_feature_pipeline(city_path: Path, log_file: Path):
    logger = setup_logger(log_file=log_file)

    buildings = load_buildings(city_path)

    with LoggingContext(logger, feature_name="Building"):
        from features import building

        buildings['FootprintArea'] = buildings.geometry.area
        buildings['Perimeter'] = buildings.geometry.length
        buildings['Phi'] = building.calculate_phi(buildings)
        buildings['LongestAxisLength'] = longest_axis_length(buildings)
        buildings['Elongation'] = elongation(buildings)
        buildings['Convexity'] = convexity(buildings)
        buildings['Orientation'] = orientation(buildings)
        buildings['Corners'] = corners(buildings)


if __name__ == "__main__":
    execute_feature_pipeline("test_data/Vaugneray", "test_data/logs/features.log")
