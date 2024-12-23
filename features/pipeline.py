import os

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from features import block, buffer, building, landuse, osm, poi, population, street, topography
from log import LoggingContext, setup_logger
from util import (
    bbox,
    building_type_harmonization,
    center,
    distance_nearest,
    load_buildings,
    load_GHS_built_up,
    load_nuts_attr,
    load_pois,
    load_streets,
    read_value,
    snearest_attr,
    store_features,
    transform_crs,
)

H3_RES = 10
H3_BUFFER_SIZES = [1, 4]  # corresponds to a buffer of 0.1 and 0.9 km^2
CRS = 3035


def execute_feature_pipeline(
    region_id: str,
    bldgs_dir: str,
    streets_dir: str,
    pois_dir: str,
    osm_bldgs_dir: str,
    built_up_path: str,
    lu_path: str,
    oceans_path: str,
    topo_path: str,
    cdd_path: str,
    hdd_path: str,
    pop_path: str,
    lau_path: str,
    out_dir: str,
    log_file: str,
) -> None:
    logger = setup_logger(log_file=log_file)

    out_file = os.path.join(out_dir, f"{region_id}.gpkg")
    if os.path.exists(out_file):
        logger.info(f"Skipping feature engineering for region {region_id} because already done.")
        return

    buildings = load_buildings(bldgs_dir, region_id)
    buildings = _preprocess(buildings)

    with LoggingContext(logger, feature_name="building"):
        buildings = _calculate_building_features(buildings)

    with LoggingContext(logger, feature_name="blocks"):
        buildings = _calculate_block_features(buildings)

    with LoggingContext(logger, feature_name="street"):
        buildings = _calculate_street_features(buildings, streets_dir, region_id)

    with LoggingContext(logger, feature_name="poi"):
        buildings = _calculate_poi_features(buildings, pois_dir, region_id)

    with LoggingContext(logger, feature_name="osm_buildings"):
        buildings = _calculate_osm_buildings_features(buildings, osm_bldgs_dir, region_id)

    with LoggingContext(logger, feature_name="GHS_built_up"):
        buildings = _calculate_GHS_built_up_features(buildings, built_up_path)

    with LoggingContext(logger, feature_name="landuse"):
        buildings = _calculate_landuse_features(buildings, lu_path, oceans_path)

    with LoggingContext(logger, feature_name="topography"):
        buildings = _calculate_topography_features(buildings, topo_path)

    with LoggingContext(logger, feature_name="climate"):
        buildings = _calculate_climate_features(buildings, cdd_path, hdd_path)

    with LoggingContext(logger, feature_name="population"):
        buildings = _calculate_population_features(buildings, pop_path)

    with LoggingContext(logger, feature_name="nuts_region"):
        buildings = _calculate_nuts_region_features(buildings, lau_path, region_id)

    with LoggingContext(logger, feature_name="buffer"):
        buildings = _calculate_building_buffer_features(buildings)

    with LoggingContext(logger, feature_name="buffer_poi"):
        buildings = _calculate_poi_buffer_features(buildings, pois_dir, region_id)

    with LoggingContext(logger, feature_name="buffer_osm_buildings"):
        buildings = _calculate_osm_buildings_buffer_features(buildings, osm_bldgs_dir, region_id)

    with LoggingContext(logger, feature_name="buffer_GHS_built_up"):
        buildings = _calculate_GHS_built_up_buffer_features(buildings, built_up_path)

    with LoggingContext(logger, feature_name="buffer_population"):
        buildings = _calculate_population_buffer_features(buildings, pop_path)

    with LoggingContext(logger, feature_name="interaction"):
        buildings = _calculate_interaction_features(buildings)

    store_features(buildings, out_dir, region_id)


def _preprocess(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = buildings.to_crs(CRS)
    buildings["h3_index"] = buffer.h3_index(buildings, H3_RES)

    type_mapping = building_type_harmonization()
    types = set(type_mapping.values())
    types.remove(np.nan)
    buildings["type"] = buildings["type_source"].map(type_mapping)
    buildings["type"] = buildings["type"].astype(CategoricalDtype(categories=types))

    buildings["height"] = buildings["height"].astype(float)
    buildings["floors"] = buildings["floors"].astype(float)
    buildings["age"] = buildings["age"].astype(float)

    return buildings


def _calculate_building_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings["bldg_lat"] = buildings.centroid.to_crs("EPSG:4326").y
    buildings["bldg_lng"] = buildings.centroid.to_crs("EPSG:4326").x
    buildings["bldg_footprint_area"] = buildings.area
    buildings["bldg_perimeter"] = buildings.length
    buildings["bldg_normalized_perimeter_index"] = building.calculate_norm_perimeter(buildings)
    buildings["bldg_area_perimeter_ratio"] = buildings["bldg_footprint_area"] / buildings["bldg_perimeter"]
    buildings["bldg_phi"] = building.calculate_phi(buildings)
    buildings["bldg_longest_axis_length"] = momepy.longest_axis_length(buildings)
    buildings["bldg_elongation"] = momepy.elongation(buildings)
    buildings["bldg_convexity"] = momepy.convexity(buildings)
    buildings["bldg_rectangularity"] = momepy.equivalent_rectangular_index(buildings)
    buildings["bldg_orientation"] = momepy.orientation(buildings)
    buildings["bldg_corners"] = momepy.corners(buildings)
    buildings["bldg_shared_wall_length"] = momepy.shared_walls(buildings)
    buildings["bldg_rel_courtyard_size"] = momepy.courtyard_area(buildings) / buildings.area
    buildings["bldg_touches"] = building.calculate_touches(buildings)
    buildings["bldg_distance_closest"] = building.calculate_distance_to_closest_building(buildings)

    return buildings


def _calculate_block_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = block.generate_blocks(buildings)
    blocks["block_length"] = blocks.building_ids.apply(len)
    blocks["block_footprint_area"] = blocks.geometry.apply(lambda g: g.area)
    blocks["block_avg_footprint_area"] = blocks.block_buildings.apply(lambda b: b.area.mean())
    blocks["block_std_footprint_area"] = blocks.block_buildings.apply(lambda b: b.area.std())
    blocks["block_perimeter"] = blocks.length
    blocks["block_longest_axis_length"] = momepy.longest_axis_length(blocks)
    blocks["block_elongation"] = momepy.elongation(blocks)
    blocks["block_convexity"] = momepy.convexity(blocks)
    blocks["block_orientation"] = momepy.orientation(blocks)
    blocks["block_corners"] = momepy.corners(blocks.convex_hull)

    buildings = block.merge_blocks_and_buildings(blocks, buildings)
    return buildings


def _calculate_street_features(buildings: gpd.GeoDataFrame, streets_dir: str, region_id: str) -> gpd.GeoDataFrame:
    streets = load_streets(streets_dir, region_id)
    streets = streets.to_crs(buildings.crs)

    buildings[["street_size", "street_distance", "street_alignment"]] = street.closest_street_features(
        buildings, streets
    )

    return buildings


def _calculate_poi_features(buildings: gpd.GeoDataFrame, pois_dir: str, region_id: str) -> gpd.GeoDataFrame:
    pois = load_pois(pois_dir, region_id)
    pois = pois.to_crs(buildings.crs)

    buildings["poi_distance"] = poi.distance_to_closest_poi(buildings, pois)

    return buildings


def _calculate_landuse_features(buildings: gpd.GeoDataFrame, lu_path: str, oceans_path: str) -> gpd.GeoDataFrame:
    buildings["lu_distance_industry"] = landuse.distance_to_landuse(buildings, "industrial", lu_path)
    buildings["lu_distance_agriculture"] = landuse.distance_to_landuse(buildings, "agricultural", lu_path)
    buildings["lu_distance_coast"] = landuse.distance_to_coast(buildings, oceans_path)

    return buildings


def _calculate_topography_features(buildings: gpd.GeoDataFrame, topo_file: str) -> gpd.GeoDataFrame:
    buildings["elevation"] = topography.calculate_elevation(buildings, topo_file)
    buildings["ruggedness"] = topography.calculate_ruggedness(buildings, topo_file, H3_RES - 2)

    return buildings


def _calculate_climate_features(buildings: gpd.GeoDataFrame, cdd_file: str, hdd_file: str) -> gpd.GeoDataFrame:
    c = center(buildings)
    c = transform_crs(c, buildings.crs, "EPSG:4326")
    lng, lat = c.x, c.y
    buildings["cdd"] = read_value(cdd_file, lng, lat)
    buildings["hdd"] = read_value(hdd_file, lng, lat)

    return buildings


def _calculate_population_features(buildings: gpd.GeoDataFrame, pop_file: str) -> gpd.GeoDataFrame:
    buildings["population"] = population.count_local_population(buildings, pop_file)

    return buildings


def _calculate_osm_buildings_features(
    buildings: gpd.GeoDataFrame, osm_bldgs_dir: str, region_id: str
) -> gpd.GeoDataFrame:
    osm_buildings = osm.load_osm_buildings(osm_bldgs_dir, region_id)
    osm_buildings = osm_buildings.to_crs(buildings.crs)

    buildings = osm.closest_building_type(buildings, osm_buildings)
    buildings["osm_closest_building_height"] = osm.closest_building_height(buildings, osm_buildings)
    buildings["osm_distance_residential"] = osm.distance_to_building_type(buildings, osm_buildings, "residential")
    buildings["osm_distance_public"] = osm.distance_to_building_type(buildings, osm_buildings, "public")
    buildings["osm_distance_industry"] = osm.distance_to_building_type(buildings, osm_buildings, "industrial")
    buildings["osm_distance_commercial"] = osm.distance_to_building_type(buildings, osm_buildings, "commercial")
    buildings["osm_distance_agriculture"] = osm.distance_to_building_type(buildings, osm_buildings, "agricultural")
    buildings["osm_distance_medium_rise"] = osm.distance_to_building_height(buildings, osm_buildings, [15, 30])
    buildings["osm_distance_high_rise"] = osm.distance_to_building_height(buildings, osm_buildings, [30, np.inf])

    return buildings


def _calculate_GHS_built_up_features(buildings: gpd.GeoDataFrame, built_up_file: str) -> gpd.GeoDataFrame:
    area = bbox(buildings, buffer=1000)
    built_up = load_GHS_built_up(built_up_file, area)
    built_up = built_up.to_crs(buildings.crs)

    nearest = snearest_attr(buildings, built_up, attr=["use_type", "height"], max_distance=100)
    nearest["use_type"] = pd.Categorical(nearest["use_type"], categories=["residential", "non-residential"])
    buildings["ghs_closest_height"] = nearest["height"]
    buildings["ghs_closest_use_type"] = nearest["use_type"]
    buildings = pd.get_dummies(buildings, columns=["ghs_closest_use_type"])

    high_rise_areas = built_up[built_up["high_rise"]]
    buildings["ghs_distance_high_rise"] = distance_nearest(buildings, high_rise_areas, max_distance=1000)

    return buildings


def _calculate_nuts_region_features(buildings: gpd.GeoDataFrame, lau_path: str, region_id: str) -> gpd.GeoDataFrame:
    nuts = load_nuts_attr(lau_path)
    region_attr = nuts.loc[region_id]

    buildings["nuts_mountain_type"] = region_attr["MOUNT_TYPE"]
    buildings["nuts_coast_type"] = region_attr["COAST_TYPE"]
    buildings["nuts_urban_type"] = region_attr["URBN_TYPE"]

    return buildings


def _calculate_building_buffer_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buffer_fts = {
        "bldg_n": ("bldg_footprint_area", "count"),
        "bldg_total_footprint_area": ("bldg_footprint_area", "sum"),
        "bldg_avg_footprint_area": ("bldg_footprint_area", "mean"),
        "bldg_std_footprint_area": ("bldg_footprint_area", "std"),
        "bldg_max_footprint_area": ("bldg_footprint_area", "max"),
        "bldg_avg_perimeter": ("bldg_perimeter", "mean"),
        "bldg_std_perimeter": ("bldg_perimeter", "std"),
        "bldg_max_perimeter": ("bldg_perimeter", "max"),
        "bldg_avg_elongation": ("bldg_elongation", "mean"),
        "bldg_std_elongation": ("bldg_elongation", "std"),
        "bldg_max_elongation": ("bldg_elongation", "max"),
        "bldg_avg_convexity": ("bldg_convexity", "mean"),
        "bldg_std_convexity": ("bldg_convexity", "std"),
        "bldg_max_convexity": ("bldg_convexity", "max"),
        "bldg_avg_orientation": ("bldg_orientation", "mean"),
        "bldg_std_orientation": ("bldg_orientation", "std"),
        "bldg_max_orientation": ("bldg_orientation", "max"),
        "bldg_avg_distance_closest": ("bldg_distance_closest", "mean"),
        "bldg_std_distance_closest": ("bldg_distance_closest", "std"),
        "bldg_max_distance_closest": ("bldg_distance_closest", "max"),
        "blocks_n": ("block_footprint_area", "count"),
        "block_avg_footprint_area": ("block_footprint_area", "mean"),
        "block_std_footprint_area": ("block_footprint_area", "std"),
        "block_max_footprint_area": ("block_footprint_area", "max"),
        "block_avg_perimeter": ("block_perimeter", "mean"),
        "block_std_perimeter": ("block_perimeter", "std"),
        "block_max_perimeter": ("block_perimeter", "max"),
        "block_avg_elongation": ("block_elongation", "mean"),
        "block_std_elongation": ("block_elongation", "std"),
        "block_max_elongation": ("block_elongation", "max"),
        "block_avg_convexity": ("block_convexity", "mean"),
        "block_std_convexity": ("block_convexity", "std"),
        "block_max_convexity": ("block_convexity", "max"),
        "block_avg_orientation": ("block_orientation", "mean"),
        "block_std_orientation": ("block_orientation", "std"),
        "block_max_orientation": ("block_orientation", "max"),
        "block_avg_corners": ("block_corners", "mean"),
        "block_std_corners": ("block_corners", "std"),
        "block_max_corners": ("block_corners", "max"),
        "block_avg_length": ("block_length", "mean"),
        "block_std_length": ("block_length", "std"),
        "block_max_length": ("block_length", "max"),
        "street_avg_distance": ("street_distance", "mean"),
        "street_std_distance": ("street_distance", "std"),
        "street_max_distance": ("street_distance", "max"),
        "street_avg_size": ("street_size", "mean"),
        "street_std_size": ("street_size", "std"),
        "street_max_size": ("street_size", "max"),
    }
    hex_grid = buffer.calculate_h3_buffer_features(buildings, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = _add_grid_fts_to_buildings(buildings, hex_grid)

    for cat, ft in [
        ("bldg", "footprint_area"),
        ("bldg", "perimeter"),
        ("bldg", "elongation"),
        ("bldg", "convexity"),
        ("bldg", "orientation"),
        ("bldg", "distance_closest"),
        ("block", "footprint_area"),
        ("block", "perimeter"),
        ("block", "elongation"),
        ("block", "convexity"),
        ("block", "orientation"),
        ("block", "corners"),
        ("block", "length"),
        ("street", "distance"),
        ("street", "size"),
    ]:
        for s in H3_BUFFER_SIZES:
            suffix = buffer.ft_suffix(H3_RES, s)
            buildings[f"{cat}_diff_{ft}_{suffix}"] = buildings[f"{cat}_avg_{ft}_{suffix}"] - buildings[f"{cat}_{ft}"]

    return buildings


def _calculate_population_buffer_features(buildings: gpd.GeoDataFrame, pop_file: str) -> gpd.GeoDataFrame:
    suffix = buffer.ft_suffix(H3_RES - 2)
    buildings[f"population_{suffix}"] = population.count_population_in_buffer(buildings, pop_file, H3_RES - 2)

    return buildings


def _calculate_poi_buffer_features(buildings: gpd.GeoDataFrame, pois_dir: str, region_id: str) -> gpd.GeoDataFrame:
    pois = load_pois(pois_dir, region_id)
    pois = pois.to_crs(buildings.crs)

    buffer_fts = {"poi_n": ("amenity", "count")}
    hex_grid = buffer.calculate_h3_buffer_features(pois, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = _add_grid_fts_to_buildings(buildings, hex_grid)

    hex_grid_large_buffer = hex_grid[hex_grid.columns[-1]]
    buildings["distance_to_center"] = buffer.distance_to_h3_grid_max(buildings, hex_grid_large_buffer)

    return buildings


def _calculate_osm_buildings_buffer_features(
    buildings: gpd.GeoDataFrame, osm_bldgs_dir: str, region_id: str
) -> gpd.GeoDataFrame:
    osm_buildings = osm.load_osm_buildings(osm_bldgs_dir, region_id)
    osm_buildings = osm_buildings.to_crs(buildings.crs)

    hex_grid_type_shares = buffer.calculate_h3_buffer_shares(osm_buildings, "type", H3_RES, H3_BUFFER_SIZES)
    hex_grid_type_shares = hex_grid_type_shares.add_prefix("osm_type_share_")
    buildings = _add_grid_fts_to_buildings(buildings, hex_grid_type_shares)

    buffer_fts = {
        "osm_avg_height": ("height", "mean"),
        "osm_std_height": ("height", "std"),
        "osm_max_height": ("height", "max"),
        "osm_type_variety": ("type", "nunique"),
    }
    hex_grid = buffer.calculate_h3_buffer_features(osm_buildings, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = _add_grid_fts_to_buildings(buildings, hex_grid)

    return buildings


def _calculate_GHS_built_up_buffer_features(buildings: gpd.GeoDataFrame, built_up_file: str) -> gpd.GeoDataFrame:
    area = bbox(buildings, buffer=1000)
    built_up = load_GHS_built_up(built_up_file, area)
    built_up = built_up.to_crs(buildings.crs)

    hex_grid_type_shares = buffer.calculate_h3_buffer_shares(built_up, "use_type", H3_RES, H3_BUFFER_SIZES)
    hex_grid_type_shares = hex_grid_type_shares.add_prefix("ghs_use_type_share_")
    buildings = _add_grid_fts_to_buildings(buildings, hex_grid_type_shares)

    # Note: could be calculated separately since classes are mutually exclusive, so likely no performance gain here
    buffer_fts = {
        "ghs_greenness": ("NDVI", "mean"),
        "ghs_height": ("height", "mean"),
    }
    hex_grid = buffer.calculate_h3_buffer_features(built_up, buffer_fts, H3_RES, H3_BUFFER_SIZES)
    buildings = _add_grid_fts_to_buildings(buildings, hex_grid)

    return buildings


def _calculate_interaction_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings["i_distance_to_built"] = buildings[["bldg_distance_closest", "street_distance"]].min(axis=1)
    suffix = buffer.ft_suffix(H3_RES, H3_BUFFER_SIZES[-1])
    buildings["i_distance_to_built_times_total_footprint_area"] = (
        buildings["bldg_distance_closest"] * buildings[f"bldg_total_footprint_area_{suffix}"]
    )
    buildings["i_population_per_footprint_area"] = (
        buildings[f"population_{buffer.ft_suffix(H3_RES - 2)}"] / buildings[f"bldg_total_footprint_area_{suffix}"]
    )

    return buildings


def _add_grid_fts_to_buildings(buildings, grid):
    return buildings.merge(grid, left_on="h3_index", right_index=True, how="left")
