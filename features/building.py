
def calculate_phi(buildings):
    max_dist = buildings.geometry.map(lambda g: g.centroid.hausdorff_distance(g.exterior))
    circle_area = buildings.geometry.centroid.buffer(max_dist).area
    return buildings.geometry.area / circle_area