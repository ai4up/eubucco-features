from log import setup_logger, LoggingContext
from util import load_buildings
from momepy import longest_axis_length, elongation, convexity, orientation, corners, shared_walls


def execute_feature_pipeline(city_path: str, log_file: str):
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
        buildings['SharedWallLength'] = shared_walls(buildings)
        buildings['Touches'] = building.calculate_touches(buildings)

    with LoggingContext(logger, feature_name="Blocks"):
        from features import block

        blocks = block.generate_blocks(buildings)
        blocks['BlockLength'] = blocks.building_ids.apply(len)
        blocks['BlockTotalFootprintArea'] = blocks.geometry.apply(lambda g: g.area)
        blocks['AvBlockFootprintArea'] = blocks.block_buildings.apply(lambda b: b.area.mean())
        blocks['StBlockFootprintArea'] = blocks.block_buildings.apply(lambda b: b.area.std())
        blocks['BlockPerimeter'] = blocks.length
        blocks['BlockLongestAxisLength'] = longest_axis_length(blocks)
        blocks['BlockElongation'] = elongation(blocks)
        blocks['BlockConvexity'] = convexity(blocks)
        blocks['BlockOrientation'] = orientation(blocks)
        blocks['BlockCorners'] = corners(blocks.convex_hull)

        buildings = block.merge_blocks_and_buildings(blocks, buildings)


if __name__ == "__main__":
    city_path = "test_data/Vaugneray"
    log_file = "test_data/logs/features.log"
    execute_feature_pipeline(city_path, log_file)
