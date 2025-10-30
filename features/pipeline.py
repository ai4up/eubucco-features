import os
from typing import Callable, Dict, Tuple

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd

from features import (
    address,
    block,
    buffer,
    building,
    builtup,
    landuse,
    neighbors,
    poi,
    population,
    region,
    satclip,
    street,
    topography
)
from log import LoggingContext, setup_logger
from util import (
    center,
    distance_to_max,
    extract_largest_polygon_from_multipolygon,
    load_buildings,
    read_value,
    store_features,
    transform_crs,
    sample_representative_validation_set_across_attributes,
)

H3_RES = 10
H3_BUFFER_SIZES = [0, 1, 4]  # corresponds to a buffer of 0.02, 0.1 and 0.9 km^2
CRS = 3035


def execute_feature_pipeline(
    region_id: str,
    bldgs_dir: str,
    addresses_path: str,
    streets_dir: str,
    pois_dir: str,
    built_up_path: str,
    lu_path: str,
    oceans_path: str,
    topo_path: str,
    cdd_path: str,
    hdd_path: str,
    pop_path: str,
    lau_path: str,
    satclip_path: str,
    out_dir: str,
    log_file: str,
) -> None:
    logger = setup_logger(log_file=log_file)

    out_file = os.path.join(out_dir, f"{region_id}.parquet")
    if os.path.exists(out_file):
        logger.info(f"Skipping feature engineering for region {region_id} because already done.")
        return

    buildings = load_buildings(bldgs_dir, region_id)
    buildings = _preprocess(buildings)

    with LoggingContext(logger, feature_name="building"):
        buildings = _calculate_building_features(buildings)

    with LoggingContext(logger, feature_name="validation_set"):
        buildings = _create_validation_set_and_mask_target_attributes(buildings)

    with LoggingContext(logger, feature_name="blocks"):
        buildings = _calculate_block_features(buildings)

    with LoggingContext(logger, feature_name="neighbors"):
        buildings = _calculate_neighbor_features(buildings)

    with LoggingContext(logger, feature_name="microsoft_heights"):
        buildings = _calculate_microsoft_height_features(buildings)

    with LoggingContext(logger, feature_name="address"):
        buildings = _calculate_address_features(buildings, addresses_path)

    with LoggingContext(logger, feature_name="street"):
        buildings = _calculate_street_features(buildings, streets_dir, region_id)

    with LoggingContext(logger, feature_name="poi"):
        buildings = _calculate_poi_features(buildings, pois_dir, region_id)
    with LoggingContext(logger, feature_name="landuse"):
        buildings = _calculate_landuse_features(buildings, lu_path, oceans_path)

    with LoggingContext(logger, feature_name="GHS_built_up"):
        buildings = _calculate_GHS_built_up_features(buildings, built_up_path)

    with LoggingContext(logger, feature_name="topography"):
        buildings = _calculate_topography_features(buildings, topo_path)

    with LoggingContext(logger, feature_name="climate"):
        buildings = _calculate_climate_features(buildings, cdd_path, hdd_path)

    with LoggingContext(logger, feature_name="population"):
        buildings = _calculate_population_features(buildings, pop_path)

    with LoggingContext(logger, feature_name="nuts_region"):
        buildings = _calculate_nuts_region_features(buildings, lau_path, region_id)

    with LoggingContext(logger, feature_name="location_encoding"):
        buildings = _calculate_location_encoding(buildings, lau_path, satclip_path, region_id)

    with LoggingContext(logger, feature_name="buffer"):
        buildings = _calculate_building_buffer_features(buildings)

    with LoggingContext(logger, feature_name="buffer_poi"):
        buildings = _calculate_poi_buffer_features(buildings, pois_dir, region_id)

    with LoggingContext(logger, feature_name="buffer_GHS_built_up"):
        buildings = _calculate_GHS_built_up_buffer_features(buildings, built_up_path)

    with LoggingContext(logger, feature_name="buffer_population"):
        buildings = _calculate_population_buffer_features(buildings, pop_path)

    with LoggingContext(logger, feature_name="interaction"):
        buildings = _calculate_interaction_features(buildings)

    buildings = _postprocess(buildings)
    store_features(buildings, out_dir, region_id)


def _preprocess(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings = buildings.to_crs(CRS)
    buildings["h3_index"] = buffer.h3_index(buildings, H3_RES)

    buildings["bldg_multi_part"] = buildings.geometry.type == "MultiPolygon"
    buildings.geometry = buildings.geometry.apply(extract_largest_polygon_from_multipolygon)

    bldgs_gt_attrs = buildings[buildings["source_dataset"].str.contains("osm|gov")]
    buildings["bldg_height"] = bldgs_gt_attrs["height"]
    buildings["bldg_age"] = bldgs_gt_attrs["age"]
    buildings["bldg_type"] = bldgs_gt_attrs["type"]
    buildings["bldg_res_type"] = bldgs_gt_attrs["residential_type"]
    buildings["bldg_msft_height"] = buildings[buildings["source_dataset"] == "msft"]["height"]
    buildings["bldg_msft_height"] = buildings["bldg_msft_height"].astype(float)

    buildings = _fill_missing_attributes_with_merged(buildings)

    return buildings


def _fill_missing_attributes_with_merged(buildings: gpd.GeoDataFrame) -> None:
    if "osm_height_merged" in buildings.columns:
        buildings["bldg_height"] = buildings["bldg_height"].fillna(buildings["osm_height_merged"])
        buildings["bldg_age"] = buildings["bldg_age"].fillna(buildings["osm_age_merged"])
        buildings["bldg_type"] = buildings["bldg_type"].fillna(buildings["osm_type_merged"])
        buildings["bldg_res_type"] = buildings["bldg_res_type"].fillna(buildings["osm_residential_type_merged"])

    if "msft_height_merged" in buildings.columns:
        buildings["bldg_msft_height"] = buildings["bldg_msft_height"].fillna(buildings["msft_height_merged"])

    return buildings


def _create_validation_set_and_mask_target_attributes(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bldg_attrs = [
        "bldg_footprint_area",
        "bldg_perimeter",
        "bldg_normalized_perimeter_index",
        "bldg_area_perimeter_ratio",
        "bldg_phi",
        "bldg_longest_axis_length",
        "bldg_elongation",
        "bldg_convexity",
        "bldg_rectangularity",
        "bldg_orientation",
        "bldg_corners",
        "bldg_shared_wall_length",
        "bldg_rel_courtyard_size",
        "bldg_touches",
        "bldg_distance_closest",
    ]
    bldgs_w_gt_attrs = buildings[buildings["source_dataset"].str.contains("osm|gov")]
    val_mask_gt = sample_representative_validation_set_across_attributes(bldgs_w_gt_attrs, ["height", "type"], bldg_attrs, val_size=0.2)
    val_mask = buildings.index.isin(bldgs_w_gt_attrs.index[val_mask_gt])

    buildings["validation"] = val_mask
    buildings.loc[val_mask, "bldg_height"] = np.nan
    buildings.loc[val_mask, "bldg_age"] = np.nan
    buildings.loc[val_mask, "bldg_type"] = np.nan
    buildings.loc[val_mask, "bldg_res_type"] = np.nan

    return buildings


def _calculate_building_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
    buildings["bldg_corners"] = momepy.corners(buildings.simplify(0.5), eps=45)
    buildings["bldg_shared_wall_length"] = momepy.shared_walls(buildings)
    buildings["bldg_rel_courtyard_size"] = momepy.courtyard_area(buildings) / buildings.area
    buildings["bldg_touches"] = building.calculate_touches(buildings)
    buildings["bldg_distance_closest"] = building.calculate_distance_to_closest_building(buildings)

    return buildings


def _calculate_block_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "block_id" not in buildings.columns:
        blocks = block.generate_blocks(buildings)
    else:
        blocks = block.generate_blocks_from_ids(buildings)

    blocks["block_length"] = blocks["building_ids"].apply(len)
    blocks["block_footprint_area"] = blocks.area
    blocks["block_perimeter"] = blocks.length
    blocks["block_normalized_perimeter_index"] = building.calculate_norm_perimeter(blocks)
    blocks["block_area_perimeter_ratio"] = blocks["block_footprint_area"] / blocks["block_perimeter"]
    blocks["block_phi"] = building.calculate_phi(blocks)
    blocks["block_longest_axis_length"] = momepy.longest_axis_length(blocks)
    blocks["block_elongation"] = momepy.elongation(blocks)
    blocks["block_convexity"] = momepy.convexity(blocks)
    blocks["block_rectangularity"] = momepy.equivalent_rectangular_index(blocks)
    blocks["block_orientation"] = momepy.orientation(blocks)
    blocks["block_corners"] = momepy.corners(blocks.simplify(0.5), eps=45)
    blocks["block_shared_wall_length"] = momepy.shared_walls(blocks)
    blocks["block_rel_courtyard_size"] = momepy.courtyard_area(blocks) / blocks.area
    blocks["block_touches"] = building.calculate_touches(blocks, id_col="block_id")
    blocks["block_distance_closest"] = building.calculate_distance_to_closest_building(blocks)

    buildings = block.merge_blocks_and_buildings(blocks, buildings)

    buildings["block_avg_footprint_area"] = buildings.groupby("block_id")["bldg_footprint_area"].transform("mean")
    buildings["block_std_footprint_area"] = buildings.groupby("block_id")["bldg_footprint_area"].transform("std")
    buildings["block_avg_perimeter"] = buildings.groupby("block_id")["bldg_perimeter"].transform("mean")
    buildings["block_std_perimeter"] = buildings.groupby("block_id")["bldg_perimeter"].transform("std")
    buildings["block_avg_elongation"] = buildings.groupby("block_id")["bldg_elongation"].transform("mean")
    buildings["block_std_elongation"] = buildings.groupby("block_id")["bldg_elongation"].transform("std")
    buildings["block_avg_orientation"] = buildings.groupby("block_id")["bldg_orientation"].transform("mean")
    buildings["block_std_orientation"] = buildings.groupby("block_id")["bldg_orientation"].transform("std")

    buildings["block_diff_footprint_area"] = buildings["block_avg_footprint_area"] - buildings["bldg_footprint_area"]
    buildings["block_diff_std_footprint_area"] = buildings["block_diff_footprint_area"] / buildings["block_std_footprint_area"]
    buildings["block_diff_perimeter"] = buildings["block_avg_perimeter"] - buildings["bldg_perimeter"]
    buildings["block_diff_std_perimeter"] = buildings["block_diff_perimeter"] / buildings["block_std_perimeter"]
    buildings["block_diff_elongation"] = buildings["block_avg_elongation"] - buildings["bldg_elongation"]
    buildings["block_diff_std_elongation"] = buildings["block_diff_elongation"] / buildings["block_std_elongation"]
    buildings["block_diff_orientation"] = buildings["block_avg_orientation"] - buildings["bldg_orientation"]
    buildings["block_diff_std_orientation"] = buildings["block_diff_orientation"] / buildings["block_std_orientation"]

    buildings = _fill_block_na_with_bldg_features(buildings)

    return buildings


def _calculate_neighbor_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings["neighbors_distance_public"] = neighbors.distance_to_building(buildings, "bldg_type", "public")
    buildings["neighbors_distance_industrial"] = neighbors.distance_to_building(buildings, "bldg_type", "industrial")
    buildings["neighbors_distance_commercial"] = neighbors.distance_to_building(buildings, "bldg_type", "commercial")
    buildings["neighbors_distance_agriculture"] = neighbors.distance_to_building(buildings, "bldg_type", "agricultural")
    buildings["neighbors_distance_residential"] = neighbors.distance_to_building(buildings, "bldg_type", "residential")
    buildings["neighbors_distance_residential_AB"] = neighbors.distance_to_building(buildings, "bldg_res_type", "apartment block")
    buildings["neighbors_distance_residential_SFH"] = neighbors.distance_to_building(buildings, "bldg_res_type", "detached single-family house")
    buildings["neighbors_distance_residential_TH"] = neighbors.distance_to_building(buildings, "bldg_res_type", "terraced house")
    buildings["neighbors_distance_residential_DH"] = neighbors.distance_to_building(buildings, "bldg_res_type", "semi-detached duplex house")
    buildings["neighbors_distance_non_residential"] = buildings[["neighbors_distance_public", "neighbors_distance_industrial", "neighbors_distance_commercial", "neighbors_distance_agriculture"]].min(axis=1)

    buildings["neighbors_closest_building_height"] = neighbors.closest_building(buildings, "bldg_height")
    buildings["neighbors_distance_low_rise"] = neighbors.distance_to_building(buildings, "bldg_height", [0, 10])
    buildings["neighbors_distance_low_medium_rise"] = neighbors.distance_to_building(buildings, "bldg_height", [10, 20])
    buildings["neighbors_distance_medium_rise"] = neighbors.distance_to_building(buildings, "bldg_height", [20, 30])
    buildings["neighbors_distance_high_rise"] = neighbors.distance_to_building(buildings, "bldg_height", [30, np.inf])

    buildings["neighbors_closest_building_age"] = neighbors.closest_building(buildings, "bldg_age")
    buildings["neighbors_distance_prior_1900"] = neighbors.distance_to_building(buildings, "bldg_age", [0, 1900])
    buildings["neighbors_distance_1900_1970"] = neighbors.distance_to_building(buildings, "bldg_age", [1900, 1970])
    buildings["neighbors_distance_1970_2000"] = neighbors.distance_to_building(buildings, "bldg_age", [1970, 2000])
    buildings["neighbors_distance_after_2000"] = neighbors.distance_to_building(buildings, "bldg_age", [2000, np.inf])

    return buildings


def _calculate_microsoft_height_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings["bldg_msft_height_closest"] = neighbors.closest_building(buildings, "bldg_msft_height")
    buildings["bldg_msft_distance_low_rise"] = neighbors.distance_to_building(buildings, "bldg_msft_height", [0, 10])
    buildings["bldg_msft_distance_low_medium_rise"] = neighbors.distance_to_building(buildings, "bldg_msft_height", [10, 20])
    buildings["bldg_msft_distance_medium_rise"] = neighbors.distance_to_building(buildings, "bldg_msft_height", [20, 30])
    buildings["bldg_msft_distance_high_rise"] = neighbors.distance_to_building(buildings, "bldg_msft_height", [30, np.inf])

    return buildings


def _calculate_address_features(buildings: gpd.GeoDataFrame, addresses_path: str) -> gpd.GeoDataFrame:
    addresses = address.load_addresses(addresses_path, buildings)

    buildings["address_count"] = address.building_address_count(buildings, addresses)
    buildings["address_unit_count"] = address.building_address_unit_count(buildings, addresses)
    buildings["address_distance"] = address.distance_to_closest_address(buildings, addresses)


    return buildings


def _calculate_street_features(buildings: gpd.GeoDataFrame, streets_dir: str, region_id: str) -> gpd.GeoDataFrame:
    streets = street.load_streets(streets_dir, region_id, CRS)

    buildings[["street_size", "street_distance", "street_alignment"]] = street.closest_street_features(
        buildings, streets
    )

    return buildings


def _calculate_poi_features(buildings: gpd.GeoDataFrame, pois_dir: str, region_id: str) -> gpd.GeoDataFrame:
    pois = poi.load_pois(pois_dir, region_id, CRS)

    buildings["poi_distance_commercial"] = poi.distance_to_closest_poi(buildings, pois, category="commercial")
    buildings["poi_distance_industrial"] = poi.distance_to_closest_poi(buildings, pois, category="industrial")
    buildings["poi_distance_education"] = poi.distance_to_closest_poi(buildings, pois, category="education")
    buildings["poi_distance_non_residential"] = buildings[["poi_distance_commercial", "poi_distance_industrial", "poi_distance_education"]].min(axis=1)

    return buildings


def _calculate_landuse_features(buildings: gpd.GeoDataFrame, lu_path: str, oceans_path: str) -> gpd.GeoDataFrame:
    lu, meta = landuse.load_landuse(lu_path, buildings)

    buildings["lu_distance_industrial"] = landuse.distance_to_landuse(buildings, lu, meta, "industrial")
    buildings["lu_distance_agriculture"] = landuse.distance_to_landuse(buildings, lu, meta, "agricultural")
    buildings["lu_distance_dense_urban"] = landuse.distance_to_landuse(buildings, lu, meta, "dense_urban")
    buildings["lu_distance_coast"] = landuse.distance_to_coast(buildings, oceans_path)

    return buildings


def _calculate_GHS_built_up_features(buildings: gpd.GeoDataFrame, built_up_file: str) -> gpd.GeoDataFrame:
    built_up, meta = builtup.load_built_up(built_up_file, buildings)

    bldg_centroids = buildings.centroid
    buildings["ghs_distance_residential"] = builtup.distance_to_ghs_class(bldg_centroids, built_up, meta, "residential")
    buildings["ghs_distance_non_residential"] = builtup.distance_to_ghs_class(bldg_centroids, built_up, meta, "non-residential")
    buildings["ghs_distance_high_rise"] = builtup.distance_to_ghs_class(bldg_centroids, built_up, meta, "high-rise")
    buildings["ghs_closest_height"] = builtup.ghs_height(bldg_centroids, built_up, meta)
    buildings["ghs_closest_height_pooled_3"] = builtup.ghs_height_pooled(bldg_centroids, built_up, meta, window_size=3)
    buildings["ghs_closest_height_pooled_5"] = builtup.ghs_height_pooled(bldg_centroids, built_up, meta, window_size=5)
    buildings["ghs_closest_height_pooled_10"] = builtup.ghs_height_pooled(bldg_centroids, built_up, meta, window_size=10)

    return buildings


def _calculate_topography_features(buildings: gpd.GeoDataFrame, topo_file: str) -> gpd.GeoDataFrame:
    elevation = topography.load_elevation(topo_file, buildings)

    buildings["elevation"] = topography.calculate_elevation(buildings, elevation)
    buildings["ruggedness"] = topography.calculate_ruggedness(buildings, elevation, H3_RES - 2)

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


def _calculate_nuts_region_features(buildings: gpd.GeoDataFrame, lau_path: str, region_id: str) -> gpd.GeoDataFrame:
    """
    Add NUTS region attributes to buildings GeoDataFrame.
    See https://ropengov.github.io/giscoR/reference/gisco_nuts.html for attribute metadata.
    """
    nuts = region.load_nuts_attr(lau_path)

    region_attr = nuts.loc[region_id]
    buildings["nuts_mountain_type"] = str(region_attr["MOUNT_TYPE"])
    buildings["nuts_coast_type"] = str(region_attr["COAST_TYPE"])
    buildings["nuts_urban_type"] = str(region_attr["URBN_TYPE"])

    buildings["nuts_mountain_type"] = pd.Categorical(buildings["nuts_mountain_type"], categories=["1", "2", "3", "4"])
    buildings["nuts_coast_type"] = pd.Categorical(buildings["nuts_coast_type"], categories=["1", "2", "3"])
    buildings["nuts_urban_type"] = pd.Categorical(buildings["nuts_urban_type"], categories=["1", "2", "3"])

    return buildings


def _calculate_location_encoding(buildings: gpd.GeoDataFrame, lau_path: str, satclip_path: str, region_id: str) -> gpd.GeoDataFrame:
    nuts = region.load_nuts_attr(lau_path)

    buildings = region.add_country(buildings, nuts, region_id)
    buildings = satclip.add_h3_embeddings(buildings, satclip_path)
    buildings["lng"] = buildings.centroid.to_crs("EPSG:4326").x
    buildings["lat"] = buildings.centroid.to_crs("EPSG:4326").y

    return buildings


def _calculate_building_buffer_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buffer_fts = {
        "bldg_n": ("bldg_footprint_area", "count"),
        "bldg_max_height": ("bldg_height", "max"),
        "bldg_min_height": ("bldg_height", "min"),
        "bldg_std_height": ("bldg_height", "std"),
        "bldg_max_age": ("bldg_age", "max"),
        "bldg_min_age": ("bldg_age", "min"),
        "bldg_std_age": ("bldg_age", "std"),
        "bldg_type_variety": ("bldg_type", "nunique"),
        "bldg_res_type_variety": ("bldg_res_type", "nunique"),
        "bldg_avg_msft_height": ("bldg_msft_height", "mean"),
        "bldg_max_msft_height": ("bldg_msft_height", "max"),
        "bldg_min_msft_height": ("bldg_msft_height", "min"),
        "bldg_std_msft_height": ("bldg_msft_height", "std"),
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
        "block_avg_rectangularity": ("block_rectangularity", "mean"),
        "block_std_rectangularity": ("block_rectangularity", "std"),
        "block_max_rectangularity": ("block_rectangularity", "max"),
        "block_avg_shared_wall_length": ("block_shared_wall_length", "mean"),
        "block_std_shared_wall_length": ("block_shared_wall_length", "std"),
        "block_max_shared_wall_length": ("block_shared_wall_length", "max"),
        "block_avg_rel_courtyard_size": ("block_rel_courtyard_size", "mean"),
        "block_std_rel_courtyard_size": ("block_rel_courtyard_size", "std"),
        "block_max_rel_courtyard_size": ("block_rel_courtyard_size", "max"),
        "block_avg_touches": ("block_touches", "mean"),
        "block_std_touches": ("block_touches", "std"),
        "block_max_touches": ("block_touches", "max"),
        "block_avg_distance_closest": ("block_distance_closest", "mean"),
        "block_std_distance_closest": ("block_distance_closest", "std"),
        "block_max_distance_closest": ("block_distance_closest", "max"),
        "block_avg_normalized_perimeter_index": ("block_normalized_perimeter_index", "mean"),
        "block_std_normalized_perimeter_index": ("block_normalized_perimeter_index", "std"),
        "block_max_normalized_perimeter_index": ("block_normalized_perimeter_index", "max"),
        "block_avg_area_perimeter_ratio": ("block_area_perimeter_ratio", "mean"),
        "block_std_area_perimeter_ratio": ("block_area_perimeter_ratio", "std"),
        "block_max_area_perimeter_ratio": ("block_area_perimeter_ratio", "max"),
        "block_avg_phi": ("block_phi", "mean"),
        "block_std_phi": ("block_phi", "std"),
        "block_max_phi": ("block_phi", "max"),
        "block_avg_length": ("block_length", "mean"),
        "block_std_length": ("block_length", "std"),
        "block_max_length": ("block_length", "max"),
        "street_avg_distance": ("street_distance", "mean"),
        "street_std_distance": ("street_distance", "std"),
        "street_max_distance": ("street_distance", "max"),
        "street_avg_size": ("street_size", "mean"),
        "street_std_size": ("street_size", "std"),
        "street_max_size": ("street_size", "max"),
        "address_total_count": ("address_count", "sum"),
        "address_avg_count": ("address_count", "mean"),
        "address_std_count": ("address_count", "std"),
        "address_max_count": ("address_count", "max"),
        "address_total_unit_count": ("address_unit_count", "sum"),
        "address_avg_unit_count": ("address_unit_count", "mean"),
        "address_std_unit_count": ("address_unit_count", "std"),
        "address_max_unit_count": ("address_unit_count", "max"),
    }
    buildings = _add_h3_buffer_features(buildings, buildings, buffer_fts)

    h3_cells = pd.DataFrame(index=buildings["h3_index"].unique())
    target_var_buffer_fts = {"bldg_avg_height": "bldg_height", "bldg_avg_age": "bldg_age"}
    buildings = buffer.add_h3_buffer_mean_excluding_self(buildings, target_var_buffer_fts, H3_RES, H3_BUFFER_SIZES, grid_cells=h3_cells)

    for s in H3_BUFFER_SIZES:
        suffix = buffer.ft_suffix(H3_RES, s)
        for cat, ft in [
            ("bldg", "age"),
            ("bldg", "height"),
            ("bldg", "msft_height"),
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
            ("block", "rectangularity"),
            ("block", "shared_wall_length"),
            ("block", "rel_courtyard_size"),
            ("block", "touches"),
            ("block", "distance_closest"),
            ("block", "normalized_perimeter_index"),
            ("block", "area_perimeter_ratio"),
            ("block", "phi"),
            ("street", "distance"),
            ("street", "size"),
            ("address", "count"),
            ("address", "unit_count"),
        ]:
            buildings[f"{cat}_diff_{ft}_{suffix}"] = buildings[f"{cat}_avg_{ft}_{suffix}"] - buildings[f"{cat}_{ft}"]
            buildings[f"{cat}_diff_std_{ft}_{suffix}"] = (buildings[f"{cat}_diff_{ft}_{suffix}"] / buildings[f"{cat}_std_{ft}_{suffix}"]).replace([np.inf, -np.inf], 0)

        buildings[f"bldg_diff_std_shape_{suffix}"] = buildings[[f"bldg_diff_std_{ft}_{suffix}" for ft in ["footprint_area", "perimeter", "elongation", "convexity", "orientation", "distance_closest"]]].abs().mean(axis=1)

    hex_grid_type_shares = buffer.calculate_h3_buffer_shares(buildings, "bldg_type", H3_RES, H3_BUFFER_SIZES, h3_cells, dropna=True, n_min=4, exclude_self=True)
    buildings = buildings.join(hex_grid_type_shares.add_prefix("bldg_type_share_"), how="left")

    hex_grid_res_type_shares = buffer.calculate_h3_buffer_shares(buildings, "bldg_res_type", H3_RES, H3_BUFFER_SIZES, h3_cells, dropna=True, n_min=4, exclude_self=True)
    buildings = buildings.join(hex_grid_res_type_shares.add_prefix("bldg_res_type_share_"), how="left")

    return buildings


def _calculate_population_buffer_features(buildings: gpd.GeoDataFrame, pop_file: str) -> gpd.GeoDataFrame:
    suffix = buffer.ft_suffix(H3_RES - 2)
    buildings[f"population_{suffix}"] = population.count_population_in_buffer(buildings, pop_file, H3_RES - 2)

    return buildings


def _calculate_poi_buffer_features(buildings: gpd.GeoDataFrame, pois_dir: str, region_id: str) -> gpd.GeoDataFrame:
    pois = poi.load_pois(pois_dir, region_id, CRS)

    buffer_fts = {"poi_n": ("amenity", "count")}
    buildings = _add_h3_buffer_features(buildings, pois, buffer_fts)

    suffix = buffer.ft_suffix(H3_RES, H3_BUFFER_SIZES[-1])
    buildings["distance_to_center"] = distance_to_max(buildings, f"poi_n_{suffix}")

    return buildings


def _calculate_GHS_built_up_buffer_features(buildings: gpd.GeoDataFrame, built_up_file: str) -> gpd.GeoDataFrame:
    bu_raster, meta = builtup.load_built_up(built_up_file, buildings)

    for size in [100, 500]:
        buildings[f"ghs_height_buffer_{size}"] = builtup.ghs_mean_height(buildings, bu_raster, meta, size)
        buildings[f"ghs_greenness_buffer_{size}"] = builtup.ghs_mean_ndvi(buildings, bu_raster, meta, size)
        buildings[f"ghs_type_share_residential_buffer_{size}"] = builtup.ghs_type_share(buildings, bu_raster, meta, size, "residential")
        buildings[f"ghs_type_share_non_residential_buffer_{size}"] = builtup.ghs_type_share(buildings, bu_raster, meta, size, "non-residential")

    return buildings


def _calculate_interaction_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings["i_distance_to_built"] = buildings[["bldg_distance_closest", "street_distance"]].min(axis=1)
    suffix = buffer.ft_suffix(H3_RES, H3_BUFFER_SIZES[-1])
    pop_suffix = buffer.ft_suffix(H3_RES - 2)

    buildings["i_distance_to_built_x_population"] = (
        buildings["bldg_distance_closest"] * buildings[f"population_{pop_suffix}"]
    )
    buildings["i_distance_to_built_x_population_x_footprint_area"] = (
        buildings["bldg_distance_closest"] * buildings[f"population_{pop_suffix}"] * np.log(buildings["bldg_footprint_area"])
    )
    buildings["i_distance_to_built_x_total_footprint_area"] = (
        buildings["bldg_distance_closest"] * (buildings[f"bldg_total_footprint_area_{suffix}"] / 1000)
    )
    buildings["i_population_per_footprint_area"] = (
        buildings[f"population_{pop_suffix}"] / (buildings[f"bldg_total_footprint_area_{suffix}"] / 1000)
    )

    return buildings


def _add_h3_buffer_features(buildings: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame, operation: Dict[str, Tuple[str, Callable]]) -> gpd.GeoDataFrame:
    h3_cells = pd.DataFrame(index=buildings["h3_index"].unique())
    hex_grid = buffer.calculate_h3_buffer_features(gdf, operation, H3_RES, H3_BUFFER_SIZES, h3_cells)
    buildings = _add_grid_fts_to_buildings(buildings, hex_grid)

    return buildings


def _add_grid_fts_to_buildings(buildings, grid):
    return buildings.merge(grid, left_on="h3_index", right_index=True, how="left")


def _fill_block_na_with_bldg_features(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fts = [
        "perimeter",
        "footprint_area",
        "normalized_perimeter_index",
        "area_perimeter_ratio",
        "phi",
        "longest_axis_length",
        "elongation",
        "convexity",
        "rectangularity",
        "orientation",
        "corners",
        "shared_wall_length",
        "rel_courtyard_size",
        "touches",
        "distance_closest",
    ]

    for ft in fts:
        buildings["block_" + ft] = buildings["block_" + ft].fillna(buildings["bldg_" + ft])

    buildings["block_length"] = buildings["block_length"].fillna(1)

    fts = [
        "perimeter",
        "footprint_area",
        "elongation",
        "orientation",
    ]
    for ft in fts:
        buildings["block_std_" + ft] = buildings["block_std_" + ft].fillna(0)
        buildings["block_avg_" + ft] = buildings["block_avg_" + ft].fillna(buildings["bldg_" + ft])
        buildings["block_diff_" + ft] = buildings["block_diff_" + ft].fillna(0)
        buildings["block_diff_std_" + ft] = buildings["block_diff_std_" + ft].replace([np.inf, -np.inf], 0)

    return buildings


def _postprocess(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fts_cols = buildings.filter(
        regex='^(bldg|block|neighbors|poi|address|street|lu|ghs|nuts|satclip|cdd|hdd|elevation|ruggedness|lat|lng|country|population|distance_to_center|i_)').columns
    buildings[fts_cols] = buildings[fts_cols].replace([None, -np.inf, np.inf], np.nan)

    return buildings
