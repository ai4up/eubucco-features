import geopandas as gpd
import numpy as np
from momepy import (
    convexity,
    corners,
    courtyard_area,
    elongation,
    equivalent_rectangular_index,
    longest_axis_length,
    orientation,
    shared_walls,
)

from features import block, buffer, building, landuse, osm, poi, population, street, topography
from log import LoggingContext, setup_logger
from util import (
    bbox,
    distance_nearest,
    load_buildings,
    load_GHS_built_up,
    load_osm_buildings,
    load_pois,
    load_streets,
    snearest_attr,
    store_features,
)

H3_RES = 10
H3_BUFFER_SIZES = [1, 4]  # corresponds to a buffer of 0.1 and 0.9 km^2


def execute_feature_pipeline(
    city_path: str, log_file: str, built_up_path: str, lu_path: str, oceans_path: str, topo_path: str, pop_path: str
) -> None:
    logger = setup_logger(log_file=log_file)

    buildings = load_buildings(city_path)
    buildings["h3_index"] = buffer.h3_index(buildings, H3_RES)

    with LoggingContext(logger, feature_name="building"):
        buildings = _calculate_building_features(buildings)

    with LoggingContext(logger, feature_name="blocks"):
        buildings = _calculate_block_features(buildings)

    with LoggingContext(logger, feature_name="street"):
        buildings = _calculate_street_features(buildings, city_path)

    with LoggingContext(logger, feature_name="poi"):
        buildings = _calculate_poi_features(buildings, city_path)

    with LoggingContext(logger, feature_name="osm_buildings"):
        buildings = _calculate_osm_buildings_features(buildings, city_path)

    with LoggingContext(logger, feature_name="GHS_built_up"):
        buildings = _calculate_GHS_built_up_features(buildings, built_up_path)

    with LoggingContext(logger, feature_name="landuse"):
        buildings = _calculate_landuse_features(buildings, lu_path, oceans_path)

    with LoggingContext(logger, feature_name="topography"):
        buildings = _calculate_topography_features(buildings, topo_path)

    with LoggingContext(logger, feature_name="population"):
        buildings = _calculate_population_features(buildings, pop_path)

    with LoggingContext(logger, feature_name="buffer"):
        buildings = _calculate_buffer_features(buildings)

    with LoggingContext(logger, feature_name="interaction"):
        buildings = _calculate_interaction_features(buildings)

    store_features(buildings, city_path)


def _calculate_building_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings["footprint_area"] = buildings.area
    buildings["perimeter"] = buildings.length
    buildings["normalized_perimeter_index"] = building.calculate_norm_perimeter(buildings)
    buildings["area_perimeter_ratio"] = buildings["footprint_area"] / buildings["perimeter"]
    buildings["phi"] = building.calculate_phi(buildings)
    buildings["longest_axis_length"] = longest_axis_length(buildings)
    buildings["elongation"] = elongation(buildings)
    buildings["convexity"] = convexity(buildings)
    buildings["rectangularity"] = equivalent_rectangular_index(buildings)
    buildings["orientation"] = orientation(buildings)
    buildings["corners"] = corners(buildings)
    buildings["shared_wall_length"] = shared_walls(buildings)
    buildings["rel_courtyard_size"] = courtyard_area(buildings) / buildings.area
    buildings["touches"] = building.calculate_touches(buildings)
    buildings["distance_to_closest_building"] = building.calculate_distance_to_closest_building(buildings)

    return buildings


def _calculate_block_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = block.generate_blocks(buildings)
    blocks["block_length"] = blocks.building_ids.apply(len)
    blocks["block_total_footprint_area"] = blocks.geometry.apply(lambda g: g.area)
    blocks["av_block_footprint_area"] = blocks.block_buildings.apply(lambda b: b.area.mean())
    blocks["st_block_footprint_area"] = blocks.block_buildings.apply(lambda b: b.area.std())
    blocks["block_perimeter"] = blocks.length
    blocks["block_longest_axis_length"] = longest_axis_length(blocks)
    blocks["block_elongation"] = elongation(blocks)
    blocks["block_convexity"] = convexity(blocks)
    blocks["block_orientation"] = orientation(blocks)
    blocks["block_corners"] = corners(blocks.convex_hull)

    buildings = block.merge_blocks_and_buildings(blocks, buildings)
    return buildings


def _calculate_buffer_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buffer_fts = {
        "avg_footprint_area": ("footprint_area", "mean"),
        "std_footprint_area": ("footprint_area", "std"),
        "max_footprint_area": ("footprint_area", "max"),
        "avg_elongation": ("elongation", "mean"),
        "std_elongation": ("elongation", "std"),
        "max_elongation": ("elongation", "max"),
        "avg_convexity": ("convexity", "mean"),
        "std_convexity": ("convexity", "std"),
        "max_convexity": ("convexity", "max"),
        "avg_orientation": ("orientation", "mean"),
        "std_orientation": ("orientation", "std"),
        "max_orientation": ("orientation", "max"),
        "avg_distance_to_closest_building": ("distance_to_closest_building", "mean"),
        "std_distance_to_closest_building": ("distance_to_closest_building", "std"),
        "max_distance_to_closest_building": ("distance_to_closest_building", "max"),
        "avg_distance_to_closest_street": ("distance_to_closest_street", "mean"),
        "std_distance_to_closest_street": ("distance_to_closest_street", "std"),
        "max_distance_to_closest_street": ("distance_to_closest_street", "max"),
        "avg_size_of_closest_street": ("size_of_closest_street", "mean"),
        "std_size_of_closest_street": ("size_of_closest_street", "std"),
        "max_size_of_closest_street": ("size_of_closest_street", "max"),
        "total_footprint_area": ("footprint_area", "sum"),
        "n_buildings": ("footprint_area", "count"),
    }
    hex_grid = buffer.calculate_h3_buffer_features(buildings, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = buildings.merge(hex_grid, left_on="h3_index", right_index=True, how="left")

    return buildings


def _calculate_street_features(buildings: gpd.GeoDataFrame, city_path: str) -> gpd.GeoDataFrame:
    streets = load_streets(city_path)

    buildings[
        ["size_of_closest_street", "distance_to_closest_street", "street_alignment"]
    ] = street.closest_street_features(buildings, streets)

    return buildings


def _calculate_poi_features(buildings: gpd.GeoDataFrame, city_path: str) -> gpd.GeoDataFrame:
    pois = load_pois(city_path)

    buildings["distance_to_closest_poi"] = poi.distance_to_closest_poi(buildings, pois)

    buffer_fts = {"n_pois": ("amenity", "count")}
    hex_grid = buffer.calculate_h3_buffer_features(pois, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = buildings.merge(hex_grid, left_on="h3_index", right_index=True, how="left")

    hex_grid_large_buffer = hex_grid[hex_grid.columns[-1]]
    buildings["distance_to_center"] = buffer.distance_to_h3_grid_max(buildings, hex_grid_large_buffer)

    return buildings


def _calculate_landuse_features(buildings: gpd.GeoDataFrame, lu_path: str, oceans_path: str) -> gpd.GeoDataFrame:
    buildings["lu_distance_to_industry"] = landuse.distance_to_landuse(buildings, "industrial", lu_path)
    buildings["lu_distance_to_agriculture"] = landuse.distance_to_landuse(buildings, "agricultural", lu_path)
    buildings["lu_distance_to_coast"] = landuse.distance_to_coast(buildings, oceans_path)

    return buildings


def _calculate_topography_features(buildings: gpd.GeoDataFrame, topo_file: str) -> gpd.GeoDataFrame:
    buildings["elevation"] = topography.calculate_elevation(buildings, topo_file)
    buildings["ruggedness"] = topography.calculate_ruggedness(buildings, topo_file, H3_RES - 2)

    return buildings


def _calculate_population_features(buildings: gpd.GeoDataFrame, pop_file: str) -> gpd.GeoDataFrame:
    buildings["population"] = population.count_local_population(buildings, pop_file)
    buildings["population_within_buffer"] = population.count_population_in_buffer(buildings, pop_file, H3_RES - 2)

    return buildings


def _calculate_osm_buildings_features(buildings: gpd.GeoDataFrame, city_path: str) -> gpd.GeoDataFrame:
    osm_buildings = load_osm_buildings(city_path)

    buildings = osm.closest_building_attributes(
        buildings, osm_buildings, {"type": "osm_closest_building_type", "height": "osm_closest_building_height"}
    )

    buildings["osm_distance_to_industry"] = osm.distance_to_some_building_type(buildings, osm_buildings, "industrial")
    buildings["osm_distance_to_commercial"] = osm.distance_to_some_building_type(
        buildings, osm_buildings, "commercial"
    )
    buildings["osm_distance_to_agriculture"] = osm.distance_to_some_building_type(
        buildings, osm_buildings, "agricultural"
    )
    buildings["osm_distance_to_education"] = osm.distance_to_some_building_type(buildings, osm_buildings, "education")

    buildings["osm_distance_to_medium_rise"] = osm.distance_to_some_building_height(buildings, osm_buildings, [15, 30])
    buildings["osm_distance_to_high_rise"] = osm.distance_to_some_building_height(
        buildings, osm_buildings, [30, np.inf]
    )

    hex_grid_type_shares = osm.building_type_share_buffer(osm_buildings, H3_RES, H3_BUFFER_SIZES)
    hex_grid_type_shares = hex_grid_type_shares.add_prefix("osm_type_share_")
    buildings = buildings.merge(hex_grid_type_shares, left_on="h3_index", right_index=True, how="left")

    buffer_fts = {
        "osm_avg_height": ("height", "mean"),
        "osm_std_height": ("height", "std"),
        "osm_max_height": ("height", "max"),
        "osm_type_variety": ("type", "nunique"),
    }
    hex_grid = buffer.calculate_h3_buffer_features(osm_buildings, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = buildings.merge(hex_grid, left_on="h3_index", right_index=True, how="left")

    return buildings


def _calculate_GHS_built_up_features(buildings: gpd.GeoDataFrame, built_up_file: str) -> gpd.GeoDataFrame:
    area = bbox(buildings, buffer=1000)
    built_up = load_GHS_built_up(built_up_file, area)
    built_up = built_up.to_crs(buildings.crs)

    nearest = snearest_attr(buildings, built_up, attr=["use_type", "height"], max_distance=100)
    buildings["GHS_nearest_use_type"] = nearest["use_type"]
    buildings["GHS_nearest_height"] = nearest["height"]

    high_rise_areas = built_up[built_up["high_rise"]]
    buildings["GHS_distance_high_rise"] = distance_nearest(buildings, high_rise_areas, max_distance=1000)

    # Note: could be calculated separately since classes are mutually exclusive, so likely no performance gain here
    buffer_fts = {
        "GHS_greenness": ("NDVI", "mean"),
        "GHS_height": ("height", "mean"),
    }
    hex_grid = buffer.calculate_h3_buffer_features(built_up, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = buildings.merge(hex_grid, left_on="h3_index", right_index=True, how="left")

    return buildings


def _calculate_interaction_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings["distance_to_closest_built_environment"] = buildings[
        ["distance_to_closest_building", "distance_to_closest_street"]
    ].min(axis=1)
    buildings["distance_to_closest_built_environment_interact_total_footprint_area"] = (
        buildings["distance_to_closest_building"] * buildings["total_footprint_area_within_0.92_buffer"]
    )
    buildings["population_per_footprint_area"] = (
        buildings["population_within_buffer"] / buildings["total_footprint_area_within_0.92_buffer"]
    )

    for ft in [
        "footprint_area",
        "elongation",
        "convexity",
        "orientation",
        "distance_to_closest_building",
        "distance_to_closest_street",
        "size_of_closest_street",
    ]:
        for buf in ["within_0.11_buffer", "within_0.92_buffer"]:
            buildings[f"deviation_{ft}_{buf}"] = buildings[f"avg_{ft}_{buf}"] - buildings[ft]

    return buildings


if __name__ == "__main__":
    city_path = "test_data/Toulouse"
    log_file = "test_data/logs/features.log"
    GHS_built_up_path = "test_data/GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10_V1_0_R4_C19.tif"
    corine_lu_path = "test_data/U2018_CLC2018_V2020_20u1.gpkg"
    oceans_path = "test_data/OSM-water-polygons.gpkg"
    topo_path = "test_data/gmted2010-mea075.tif"
    GHS_pop_path = "test_data/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif"

    execute_feature_pipeline(
        city_path, log_file, GHS_built_up_path, corine_lu_path, oceans_path, topo_path, GHS_pop_path
    )
