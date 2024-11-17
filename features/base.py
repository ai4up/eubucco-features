from datetime import datetime
from typing import Optional
from pathlib import Path


class Feature:
    name: str = None
    file_appendix: str = None

    def __init__(self, city_path: str):
        self.city_path = city_path
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        if not self.name:
            raise ValueError("Feature name is not set")

        if not self.file_appendix:
            raise ValueError("File appendix is not set")

    def file_name(self) -> str:
        return f"{self.name}_{self.file_appendix}.csv"

    def file_path(self) -> Path:
        return Path(f"{self.city_path}/{self.file_name()}")

    def feature_exists(self) -> bool:
        if self.file_path().is_file():
            return True
        return self.file_path().is_file()

    def _create_feature(self):
        raise NotImplementedError

    def create_feature(self):
        if self.feature_exists():
            return

        self._start_time = datetime.now()
        self._create_feature()
