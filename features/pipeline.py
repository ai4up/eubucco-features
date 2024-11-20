from pathlib import Path

from log import setup_logger, LoggingContext
from util import load_buildings
from to_refactor.momepy_functions import momepy_LongestAxisLength, momepy_Elongation, momepy_Convexeity, \
    momepy_Orientation, momepy_Corners


def execute_feature_pipeline(city_path: Path, log_file: Path):
    logger = setup_logger(log_file=log_file)

    buildings = load_buildings(city_path)

    with LoggingContext(logger, feature_name="Building"):
        from features import building

        buildings['FootprintArea'] = buildings.geometry.area
        buildings['Perimeter'] = buildings.geometry.length
        buildings['Phi'] = building.calculate_phi(buildings)
        buildings['LongestAxisLength'] = momepy_LongestAxisLength(buildings).series
        buildings['Elongation'] = momepy_Elongation(buildings).series
        buildings['Convexity'] = momepy_Convexeity(buildings).series
        buildings['Orientation'] = momepy_Orientation(buildings).series
        buildings['Corners'] = momepy_Corners(buildings).series


if __name__ == "__main__":
    execute_feature_pipeline("test_data/Vaugneray", "test_data/logs/features.log")
