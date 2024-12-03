import geopandas as gpd
import networkx as nx


def generate_blocks(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    working_copy = buildings[['id', 'geometry']]
    touching = gpd.sjoin(working_copy, working_copy, predicate='touches')

    graph = nx.Graph()
    graph.add_edges_from(zip(touching['id_left'], touching['id_right']))

    connected_components = list(nx.connected_components(graph))

    working_copy.set_index('id', inplace=True)

    blocks = []
    for component in connected_components:
        block_buildings = working_copy.loc[list(component), 'geometry']
        block_geometry = block_buildings.union_all()
        block_indices = list(component)
        blocks.append(
            {'geometry': block_geometry, 'building_ids': block_indices, 'block_buildings': block_buildings.values})

    blocks_gdf = gpd.GeoDataFrame(blocks, geometry='geometry', crs=buildings.crs)

    return blocks_gdf


def merge_blocks_and_buildings(blocks: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = blocks.drop(columns=['geometry', 'block_buildings'])
    blocks = blocks.explode('building_ids')
    buildings = buildings.merge(blocks, left_on='id', right_on='building_ids', how='left')

    return buildings
