import os
import sys

PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, "..", ".."))
sys.path.append(PROJECT_SRC_PATH)

from features.pipeline import execute_feature_pipeline  # noqa: E402


def test_pipeline():
    city_path = os.path.join(PROJECT_SRC_PATH, "tests", "data", "Vaugneray")
    log_file = os.path.join(PROJECT_SRC_PATH, "tests", "logs", "features.log")
    GHS_built_up_path = os.path.join(city_path, "GHS_BUILT_test_region.tif")
    corine_lu_path = os.path.join(city_path, "CORINE_landuse_test_region.gpkg")
    oceans_path = os.path.join(city_path, "OSM_oceans_test_region.gpkg")
    topo_path = os.path.join(city_path, "GMTED_topography_test_region.tif")
    cdd_path = os.path.join(city_path, "CDD_historical_mean_v1.nc")
    hdd_path = os.path.join(city_path, "HDD_historical_mean_v1.nc")
    GHS_pop_path = os.path.join(city_path, "GHS_POP_test_region.tif")

    execute_feature_pipeline(
        city_path,
        log_file,
        GHS_built_up_path,
        corine_lu_path,
        oceans_path,
        topo_path,
        cdd_path,
        hdd_path,
        GHS_pop_path,
    )
