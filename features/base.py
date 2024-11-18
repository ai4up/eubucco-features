from datetime import datetime
from typing import Optional, List
from pathlib import Path

import pandas as pd
from shapely import wkt

import logging

from geopandas import GeoDataFrame
import geopandas as gpd

from log import ContextFormatter, StatsLogger

CRS_UNI = 'EPSG:3035'


class Feature:
    name: str = None
    file_appendix: str = None
    comments: List[str] = []
    buildings: GeoDataFrame = None

    def __init__(self, city_path: str, log_dir: Optional[str] = None):
        self.city_path = city_path
        self.city_name = city_path.split("/")[-1]
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        if not self.name:
            raise ValueError("Feature name is not set")

        if not self.file_appendix:
            raise ValueError("File appendix is not set")

        self.logger = self._setup_logger(log_dir)
        self.stats_logger = StatsLogger(log_path=log_dir)

    def _setup_logger(self, log_dir: Optional[str]) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # If root logger propagation is enabled, logs from other modules will also be captured
        logger.propagate = False  # Prevent duplicate logging if parent loggers also have handlers

        if log_dir:
            log_path = Path(log_dir) / f"features.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)

            # Use custom formatter with city_path and feature_name context
            formatter = ContextFormatter(
                fmt="%(asctime)s - %(levelname)s - %(city_path)s - %(feature_name)s - %(message)s",
                city_path=self.city_path,
                feature_name=self.name
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def file_name(self) -> str:
        return f"{self.name}_{self.file_appendix}.csv"

    def file_path(self) -> Path:
        return Path(f"{self.city_path}/{self.file_name()}")

    def feature_exists(self) -> bool:
        exists = self.file_path().is_file()
        if exists:
            self.logger.info(f"Feature file exists: {self.file_path()}")
        else:
            self.logger.info(f"Feature file does not exist: {self.file_path()}")
        return exists

    def count_buildings(self) -> int:
        return len(self.buildings.index)

    def comment(self, comment: str):
        self.comments.append(comment)

    def _log_stats(self):
        self._end_time = datetime.now()
        self.stats_logger.log(
            city=self.city_path,
            feature=self.name,
            start_time=self._start_time,
            end_time=self._end_time,
            duration=self._end_time - self._start_time,
            buildings=self.count_buildings(),
            comments="\n".join(self.comments)
        )

    def _create_feature(self):
        raise NotImplementedError

    def _ingest_buildings(self):
        df = pd.read_csv(f"{self.city_path}/{self.city_name}_geom.csv")
        self.buildings = gpd.GeoDataFrame(
            df,
            geometry=df['geometry'].apply(wkt.loads),
            crs=CRS_UNI
        )

    def _save_feature(self):
        self.buildings.to_csv(self.file_path(), index=False)

    def create_feature(self):
        if self.feature_exists():
            return

        self._start_time = datetime.now()
        self._ingest_buildings()
        self._create_feature()
        self._save_feature()
        self._log_stats()

