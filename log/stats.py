import csv

from filelock import FileLock


class StatsLogger:
    fieldnames = ["city", "feature", "start_time", "end_time", "duration", "buildings", "comments"]

    def __init__(self, log_path):
        self.lock_path = log_path
        self.log_file = f"{log_path}/stats.csv"
        self.lock = FileLock(f"{log_path}/stats.lock")
        self._initialize_csv()

    def _initialize_csv(self):
        try:
            with open(self.log_file, mode="x", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()  # Write the header if the file is created
        except FileExistsError:
            pass  # File already exists, no need to write headers again

    def log(self, city, feature, start_time, end_time, duration, buildings, comments):
        data = {
            "city": city,
            "feature": feature,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "buildings": buildings,
            "comments": comments,
        }
        with self.lock:
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writerow(data)
