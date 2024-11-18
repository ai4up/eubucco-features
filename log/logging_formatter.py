import logging


class ContextFormatter(logging.Formatter):
    """
    Custom formatter to include additional context (e.g., city_path and feature name) in log messages.
    """

    def __init__(self, fmt: str = None, city_path: str = "", feature_name: str = ""):
        super().__init__(fmt)
        self.city_path = city_path
        self.feature_name = feature_name

    def format(self, record: logging.LogRecord) -> str:
        # Add city_path and feature_name to the log message
        record.city_path = self.city_path
        record.feature_name = self.feature_name
        return super().format(record)
