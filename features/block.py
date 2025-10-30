import uuid

import geopandas as gpd
import networkx as nx

from util import extract_largest_polygon_from_multipolygon, simplified_rectangular_buffer

def generate_blocks(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = buildings[["geometry"]]
    touching = gpd.sjoin(geom, geom, predicate="intersects")
    touching = touching[touching.index != touching["index_right"]]

    graph = nx.Graph()
    graph.add_edges_from(zip(touching.index, touching["index_right"]))
    connected_components = list(nx.connected_components(graph))

    blocks = []
    for component in connected_components:
        block_buildings = buildings.loc[list(component)]
        blocks.append(
            {
                "geometry": block_buildings["geometry"].union_all(),
                "building_ids": block_buildings["id"].values,
                "block_id": uuid.uuid4().hex[:16],
            }
        )

    if blocks:
        blocks_gdf = gpd.GeoDataFrame(blocks, geometry="geometry", crs=buildings.crs)
        blocks_gdf.geometry = simplified_rectangular_buffer(blocks_gdf, 0.01)  # ensure all geometries are Polygons and valid
        blocks_gdf.geometry = blocks_gdf.geometry.apply(extract_largest_polygon_from_multipolygon)
    else:
        blocks_gdf = gpd.GeoDataFrame(columns=["geometry", "building_ids", "block_id"])

    print(
        f"Generated {len(blocks_gdf)} blocks with on average "
        f"{blocks_gdf['building_ids'].apply(len).mean():.1f} buildings."
    )

    return blocks_gdf


def generate_blocks_from_ids(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Generate blocks from building's predefined block ids.
    """
    blocks_gdf = (
        buildings[["id", "block_id", "geometry"]]
        .dissolve(by="block_id", aggfunc={"id": list})
        .rename(columns={"id": "building_ids"})
        .reset_index()
    )

    blocks_gdf.geometry = simplified_rectangular_buffer(blocks_gdf, 0.01)  # ensure all geometries are Polygons and valid
    blocks_gdf.geometry = blocks_gdf.geometry.apply(extract_largest_polygon_from_multipolygon)

    return blocks_gdf


def merge_blocks_and_buildings(blocks: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = (
        blocks
        .drop(columns=["geometry"])
        .explode("building_ids")
    )
    buildings = (
        buildings
        .drop(columns=["block_id"], errors="ignore")
        .merge(blocks, left_on="id", right_on="building_ids", how="left")
        .drop(columns="building_ids")
    )

    return buildings
