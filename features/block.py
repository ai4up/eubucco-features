import momepy
import geopandas as gpd
import networkx as nx
import pandas as pd


def calculate_tesselation(buildings):
    limit = momepy.buffered_limit(buildings, buffer=100)
    tessellation = momepy.morphological_tessellation(buildings, clip=limit)
    return tessellation


def generate_blocks(buildings) -> gpd.GeoDataFrame:
    working_copy = buildings[['id', 'geometry']].copy(deep=True)
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


def merge_blocks_and_buildings(blocks, buildings):
    block_mapping = []
    for block_id, row in blocks.iterrows():
        for building_index in row['building_ids']:
            block_mapping.append({'building_index': building_index, 'block_id': block_id})

    block_mapping_df = pd.DataFrame(block_mapping)

    block_mapping_df = block_mapping_df.merge(blocks.drop(columns=['geometry', 'building_ids']),
                                              left_on='block_id',
                                              right_index=True)

    buildings = buildings.merge(block_mapping_df, left_on='id', right_on='building_index', how='left')
    buildings.drop(columns=['building_index', 'block_id', 'block_buildings'], inplace=True)
    return buildings
