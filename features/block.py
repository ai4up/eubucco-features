import uuid

import geopandas as gpd
import networkx as nx


def generate_blocks(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = buildings[["geometry"]]
    touching = gpd.sjoin(geom, geom, predicate="intersects")
    touching = touching[touching.index != touching["index_right"]]

    graph = nx.Graph()
    graph.add_edges_from(zip(touching.index, touching["index_right"]))
    connected_components = list(nx.connected_components(graph))

    blocks = []
    for component in connected_components:
        block_buildings = buildings.loc[list(component), "geometry"]
        block_geometry = block_buildings.union_all()
        building_ids = buildings.loc[list(component), "id"]
        blocks.append(
            {
                "geometry": block_geometry,
                "building_ids": building_ids.values,
                "block_buildings": block_buildings.values,
                "block_id": uuid.uuid4().hex[:16],
            }
        )

    blocks_gdf = gpd.GeoDataFrame(blocks, geometry="geometry", crs=buildings.crs)
    print(
        f"Generated {len(blocks_gdf)} blocks with on average "
        f"{blocks_gdf['building_ids'].apply(len).mean():.1f} buildings."
    )

    return blocks_gdf


def merge_blocks_and_buildings(blocks: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = blocks.drop(columns=["geometry", "block_buildings"])
    blocks = blocks.explode("building_ids")
    buildings = buildings.merge(blocks, left_on="id", right_on="building_ids", how="left")
    buildings = buildings.drop(columns="building_ids")

    return buildings
