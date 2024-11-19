# This is an example feature that can be used as a template for new features
from features.base import Feature
from to_refactor.momepy_functions import momepy_LongestAxisLength, momepy_Elongation, momepy_Convexeity, \
    momepy_Orientation, momepy_Corners


class BuildingFeature(Feature):
    name = "building_feature"
    file_appendix = "bld_fts"

    def _calculate_phi(self):
        max_dist = self.buildings.geometry.map(lambda g: g.centroid.hausdorff_distance(g.exterior))
        circle_area = self.buildings.geometry.centroid.buffer(max_dist).area
        return self.buildings.geometry.area / circle_area

    def _create_feature(self):
        self.buildings['FootprintArea'] = self.buildings.geometry.area
        self.buildings['Perimeter'] = self.buildings.geometry.length
        self.buildings['Phi'] = self._calculate_phi()
        self.buildings['LongestAxisLength'] = momepy_LongestAxisLength(self.buildings).series
        self.buildings['Elongation'] = momepy_Elongation(self.buildings).series
        self.buildings['Convexity'] = momepy_Convexeity(self.buildings).series
        self.buildings['Orientation'] = momepy_Orientation(self.buildings).series
        self.buildings['Corners'] = momepy_Corners(self.buildings).series
        # todo add Touches


if __name__ == "__main__":
    feature = BuildingFeature(city_path="test_data/Vaugneray", log_dir="test_data/logs", overwrite=True)
    feature.create_feature()

    print(feature.buildings.head())
