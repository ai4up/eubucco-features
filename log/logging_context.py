from datetime import datetime
import logging


class LoggingContext:
    def __init__(self, logger: logging.Logger, feature_name: str):
        self.logger = logger
        self.feature_name = feature_name
        self.start_time: datetime = None
        self.filter = self._create_filter()

    def __enter__(self):
        # Add the pre-created filter dynamically
        self.logger.addFilter(self.filter)
        self.start_time = datetime.now()
        self.logger.info(f"Entering context for feature: {self.feature_name}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Exiting context for feature: {self.feature_name}. Duration: {duration}")

        # Log any exceptions that occurred
        if exc_type:
            self.logger.error(f"An error occurred: {exc_value}", exc_info=(exc_type, exc_value, traceback))

        # Remove the filter to clean up
        self.logger.removeFilter(self.filter)

    def _create_filter(self):
        """Creates a filter that dynamically injects the feature_name into log records."""
        class FeatureNameFilter(logging.Filter):
            def __init__(self, feature_name: str):
                super().__init__()
                self.feature_name = feature_name

            def filter(self, record):
                record.feature_name = self.feature_name
                return True

        # Provide the `feature_name` to the filter
        return FeatureNameFilter(self.feature_name)
