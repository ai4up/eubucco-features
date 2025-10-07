import os
import sys
from pathlib import Path

PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, "..", ".."))
sys.path.append(PROJECT_SRC_PATH)

from features.pipeline import execute_feature_pipeline  # noqa: E402


def test_pipeline():
    region_id = "Vaugneray"
    test_dir = os.path.join(PROJECT_SRC_PATH, "tests")
    test_data_dir = os.path.join(test_dir, "data")
    bldgs_dir = os.path.join(test_data_dir, "bldgs-881f902143fffff")
    streets_dir = os.path.join(test_data_dir, "streets")
    pois_dir = os.path.join(test_data_dir, "pois")
    GHS_built_up_path = os.path.join(test_data_dir, "GHS_BUILT_test_region.tif")
    corine_lu_path = os.path.join(test_data_dir, "CORINE_landuse_test_region.tif")
    oceans_path = os.path.join(test_data_dir, "OSM_oceans_test_region.gpkg")
    topo_path = os.path.join(test_data_dir, "GMTED_topography_test_region.tif")
    cdd_path = os.path.join(test_data_dir, "CDD_historical_mean_v1.nc")
    hdd_path = os.path.join(test_data_dir, "HDD_historical_mean_v1.nc")
    GHS_pop_path = os.path.join(test_data_dir, "GHS_POP_test_region.tif")
    lau_path = os.path.join(test_data_dir, "NUTS_LAU_attr_test_region.csv")
    satclip_path = os.path.join(test_data_dir, "satclip_res8_pca64_test_region.parquet")
    out_dir = test_dir
    log_file = os.path.join(test_dir, "logs", "features.log")
    out_file = os.path.join(out_dir, f"{region_id}.parquet")

    Path(out_file).unlink(missing_ok=True)
    execute_feature_pipeline(
        region_id,
        bldgs_dir,
        streets_dir,
        pois_dir,
        GHS_built_up_path,
        corine_lu_path,
        oceans_path,
        topo_path,
        cdd_path,
        hdd_path,
        GHS_pop_path,
        lau_path,
        satclip_path,
        out_dir,
        log_file,
    )
