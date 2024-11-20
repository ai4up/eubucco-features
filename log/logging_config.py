import logging
import os
from pathlib import Path


class DefaultFeatureNameFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "feature_name"):
            record.feature_name = "no-feature"
        return True


def setup_logger(log_file: Path = None) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(feature_name)s - %(message)s"
    )

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # File handler for logging to a file
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create directory if it doesn't exist

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger("feature_engineering")
    logger.setLevel(logging.INFO)

    # Add handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Add the filter to inject default 'feature_name'
    logger.addFilter(DefaultFeatureNameFilter())

    return logger