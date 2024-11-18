# This is an example feature that can be used as a template for new features
from features.base import Feature
from time import sleep


class ExampleFeature(Feature):
    name = "example_feature"
    file_appendix = "example"

    def _create_feature(self):
        self.logger.info("Creating feature")
        sleep(5)
        print(self.buildings.head())
        self.comment("This is a comment")
        self.logger.info("Feature created")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    feature = ExampleFeature(city_path="test_data/Vaugneray", log_dir="test_data/logs")
    feature.create_feature()

    feature.buildings.plot()
    plt.show()
