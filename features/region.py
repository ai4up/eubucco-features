import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


def load_nuts_attr(lau_path: str) -> pd.DataFrame:
    nuts = pd.read_csv(lau_path)
    nuts = nuts.drop_duplicates(subset=["NUTS_ID_3"])
    nuts = nuts.set_index("NUTS_ID_3")
    nuts_attr = ["MOUNT_TYPE", "COAST_TYPE", "URBN_TYPE"]
    nuts[nuts_attr] = nuts[nuts_attr].replace(0, np.nan)

    return nuts


def add_country(buildings: gpd.GeoDataFrame, nuts: pd.DataFrame, region_id: str) -> gpd.GeoDataFrame:
    countries = nuts["CNTR_CODE"].unique()
    buildings["country"] = nuts.loc[region_id]["CNTR_CODE"]
    buildings["country"] = buildings["country"].astype(CategoricalDtype(categories=countries))

    return buildings
