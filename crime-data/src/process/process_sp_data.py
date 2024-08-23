import argparse
import datetime
import json
import os
import pathlib
import pickle

import fiona
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon
from utils import process_station_data

# from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite data.",
)
parser.add_argument(
    "--process_id",
    "-id",
    default=None,
    type=str,
    help="Id to idenfity processed data. This is used as the folder name to save files. If None, use datetime.now()",
)
parser.add_argument(
    "--network_id",
    "-net",
    default="all_private",
    type=str,
    help="Which type of graph to get from osmnx. Default is drive, but using all also makes sense.",
)
parser.add_argument(
    "--region",
    "-r",
    default=None,
    type=str,
    help="Region to donwnload using Osmnx function get_from_place.",
)
parser.add_argument(
    "--polygon",
    "-pol",
    default="sp-centro",
    type=str,
    help="Key name in polygons_coordinates.json file containing polygon coordinates to use in osmns get_from_polygon.",
)
parser.add_argument(
    "--max_distance",
    "-md",
    default=50,
    type=float,
    help="Ignore crimes with greater distance to the closest street segment than max_distance.",
)

args = parser.parse_args()
dict_args = vars(parser.parse_args())
dict_args = {str(k): str(v) for k, v in dict_args.items()}

dict_args["process_date"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if args.process_id is None:
    folder_name = dict_args["process_date"]
else:
    folder_name = args.process_id


def ensure_list(obj):
    return [obj] if isinstance(obj, str) else obj


crime_data_path = pathlib.Path("data/raw/sp/")
crime_data_path.mkdir(parents=True, exist_ok=True)

crime_processed_path = pathlib.Path(f"data/processed/sp/{folder_name}")
crime_processed_path.mkdir(parents=True, exist_ok=True)

with open(crime_processed_path / "parameters.json", "w") as f:
    json.dump(dict_args, f, sort_keys=True, indent=4)

assert (crime_data_path / "crimes.h5").is_file(), "Missing crimes.h5 file. Run the download script before!"

print("Reading crimes.h5 file...")
df_crimes = pd.read_hdf(crime_data_path / "crimes.h5")

# for column in ["latitude", "longitude"]:
#    df_crimes[column] = df_crimes[column].str.replace(",", ".").astype(float)

df_crimes = df_crimes[df_crimes.latitude < 0].copy()
print("Done!\n")


print("Creating graph using osmnx... this may take a couple of minutes...")

if args.polygon is None:
    region = args.region
else:
    with open("src/process/polygons_coordinates.json") as json_file:
        polygon_coordinates_json = json.load(json_file)

    assert (
        args.polygon in polygon_coordinates_json.keys()
    ), f"Invalid Polygon name. The possible polygons are: {list(polygon_coordinates_json.keys())}"

    polygon = polygon_coordinates_json[args.polygon]
    region = Polygon(polygon)


if not (crime_processed_path / "raw_graph.pickle").is_file() or args.overwrite:
    ox.settings.use_cache = True
    if args.polygon is None:
        graph = ox.graph_from_place(region, network_type=args.network_id)
    else:
        graph = ox.graph_from_polygon(region, network_type=args.network_id)

    highways_to_keep = [
        "secondary",
        "residential",
        "primary",
        "tertiary",
        "trunk",
        "primary_link",
        "living_street",
        "trunk_link",
        "secondary_link",
        "tertiary_link",
        "pedestrian",
    ]

    # graph = ox.graph_from_polygon(region, network_type=args.network_id)
    edges_to_remove = [
        (i1, i2)
        for i1, i2, e in graph.edges(data=True)
        if len(set(ensure_list(e["highway"])).intersection(set(highways_to_keep))) == 0
    ]

    graph.remove_edges_from(edges_to_remove)

    with open(crime_processed_path / "raw_graph.pickle", "wb") as f:
        pickle.dump(graph, f)
else:
    with open(crime_processed_path / "raw_graph.pickle", "rb") as f:
        graph = pickle.load(f)

print("Done!\n")


print("Creating undirected graph... this may take a couple of minutes...")

if not (crime_processed_path / "graph-crossing_as_nodes.pickle").is_file() or args.overwrite:
    uni_graph = ox.convert.to_undirected(graph)

    with open(crime_processed_path / "graph-crossing_as_nodes.pickle", "wb") as f:
        pickle.dump(uni_graph, f)
else:
    with open(crime_processed_path / "graph-crossing_as_nodes.pickle", "rb") as f:
        uni_graph = pickle.load(f)

print("Done!\n")


print("Inverting nodes and edges...")
resave_graph = False
if not (crime_processed_path / "graph-streets_as_nodes.pickle").is_file() or args.overwrite:
    resave_graph = True
    inv_graph = nx.line_graph(uni_graph)

    # Verify if Graph is disjoint or has subgraphs
    if not nx.is_connected(inv_graph):

        # Get the disconnected Subgraphs and select the Graph with Maximum number of nodes
        subgraphs = [inv_graph.subgraph(nodes) for nodes in nx.connected_components(inv_graph)]
        inv_graph = nx.Graph(max(subgraphs, key=lambda subgraph: len(subgraph)))

    for u, v, k in inv_graph.nodes():
        data = dict()
        data = uni_graph.edges[u, v, k]
        edge_attributes = {"source": u, "target": v, "key": k, **data}
        inv_graph.nodes[(u, v, k)].update(edge_attributes)

    inv_graph.graph["crs"] = uni_graph.graph["crs"]

    with open(crime_processed_path / "graph-streets_as_nodes.pickle", "wb") as f:
        pickle.dump(inv_graph, f)
else:
    with open(crime_processed_path / "graph-streets_as_nodes.pickle", "rb") as f:
        inv_graph = pickle.load(f)

        # Verify if Graph is disjoint or has subgraphs
        if not nx.is_connected(inv_graph):

            # Get the disconnected Subgraphs and select the Graph with Maximum number of nodes
            subgraphs = [inv_graph.subgraph(nodes) for nodes in nx.connected_components(inv_graph)]
            inv_graph = nx.Graph(max(subgraphs, key=lambda subgraph: len(subgraph)))
print("Done!\n")


print("Processing data...")

# TODO: Process data only if it does not exist

nodes_data = [data for node, data in inv_graph.nodes(data=True)]
osmnx_segments = gpd.GeoDataFrame(nodes_data)
osmnx_segments.crs = uni_graph.graph["crs"]

# Reindex the graph
if resave_graph:
    dict_index = {(row['source'], row['target'], row['key']): index for index, row in osmnx_segments.iterrows()}
    nx.relabel_nodes(inv_graph, dict_index, copy = False)
    
    with open(crime_processed_path / "graph-streets_as_nodes.pickle", "wb") as f:
        pickle.dump(inv_graph, f)

with open(crime_processed_path / "street_nodes_dataframe.pickle", "wb") as f:
    pickle.dump(osmnx_segments, f)

nodes = osmnx_segments[["geometry", "length", "highway"]]
nodes = nodes.reset_index(drop=True)

geometry_crimes = [Point(xy) for xy in zip(df_crimes.longitude, df_crimes.latitude)]
geometry_crimes = pd.Series(geometry_crimes, index=df_crimes.index)
geometry_edges = nodes["geometry"].copy()

# Create GeoDataFrames using geometries

gdf_crimes = gpd.GeoDataFrame(df_crimes, geometry=geometry_crimes)
gdf_crimes["crime_coord"] = geometry_crimes
gdf_edges = gpd.GeoDataFrame(pd.DataFrame(nodes), geometry=geometry_edges)

gdf_crimes.crs = geometry_edges.crs
gdf_crimes = gdf_crimes.to_crs("EPSG:31983")  # Convert to EPSG 31983

gdf_edges["street_segment"] = gdf_edges["geometry"]
gdf_edges = gdf_edges.to_crs("EPSG:31983")  # Convert to EPSG 31983
gdf_edges["street_segment_31983"] = gdf_edges.geometry


# Processing points of interest (geosampa)
print("Processing points of interest...")

fiona.drvsupport.supported_drivers["kml"] = "rw"  # enable KML support
fiona.drvsupport.supported_drivers["KML"] = "rw"
fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
for pasta in os.listdir(crime_data_path / "geosampa/"):
    abdf = []
    res = []
    for dir_path, dir_names, file_names in os.walk(crime_data_path / ("geosampa/" + pasta)):
        for file in file_names:
            res.append(os.path.join(dir_path, file))
    for file in res:
        if file.endswith(".shp") or file.endswith(".kml"):
            if file.endswith(".shp"):
                aux = gpd.read_file(file)
            else:
                aux = gpd.read_file(file, drive="KML")
            if aux.crs is None:
                aux = aux.set_crs("EPSG:31983")
            else:
                aux = aux.to_crs("EPSG:31983")
            # Verificar se é um ponto, se for polygon ignora
            # Assumindo que se a primeira linha for ponto, todas são
            if "Point" in str(type(aux.geometry[0])):
                abdf.append(aux)

    if abdf != []:
        abdf = gpd.GeoDataFrame(pd.concat(abdf, ignore_index=True))
        abdf.geometry = abdf.geometry.buffer(222)
        gdf_edges[pasta] = 0
        counts = gpd.sjoin(gdf_edges.to_crs(abdf.crs), abdf).index.value_counts().sort_index()
        gdf_edges.loc[counts.index, pasta] = counts.values

with open(crime_processed_path / "gdf_edges.pickle", "wb") as f:
    pickle.dump(gdf_edges, f)

print("Done!")

# Processing weather data
print("Processing weather data...")

station_names = ["barueri", "sao paulo - interlagos", "sao paulo - mirante"]

for station in station_names:
    wdata, coords = process_station_data(station, crime_data_path)

    # save temporal station data
    with open(crime_processed_path / (station + ".pickle"), "wb") as f:
        pickle.dump(wdata, f)

    # For each node, get distance to station
    gdf_station = gpd.GeoDataFrame(geometry=gpd.points_from_xy([coords[1]], [coords[0]]), crs="EPSG:4326")
    gdf_station = gdf_station.to_crs("EPSG:31983")
    gdf_edges["distance_to" + station.split(" ")[-1]] = gdf_edges.distance(gdf_station["geometry"].iloc[0])

with open(crime_processed_path / "gdf_edges.pickle", "wb") as f:
    pickle.dump(gdf_edges, f)

print("Done!")

if args.polygon is None:
    # Remove crimes outside administrative boundaries

    limites = gpd.read_file(crime_data_path / "limites_administrativos.shp")
    limites.crs = "EPSG:31983"
    limites_latlong = limites.to_crs(geometry_edges.crs)

    gdf_crimes = gpd.sjoin(gdf_crimes, limites, how="inner", predicate="within").drop(
        columns="index_right",
    )

else:
    # Remove crimes outside Polygon

    index_keep_crimes = gpd.sjoin(
        gpd.GeoDataFrame(geometry=geometry_crimes),
        gpd.GeoDataFrame(geometry=pd.Series(region)),
        how="inner",
        predicate="intersects",
    ).index
    gdf_crimes = gdf_crimes.loc[index_keep_crimes]


# For each crime, get the nearest street segment

crime_nodes = gpd.sjoin_nearest(
    gdf_crimes,
    gdf_edges,
    distance_col="distance",
    how="left",
)

# Nos casos equidistantes, pegar apenas o maior segmento de rua
crime_nodes = crime_nodes.sort_values(by="length", ascending=False)
crime_nodes = crime_nodes[~crime_nodes.index.duplicated()]
crime_nodes = crime_nodes.sort_index()

if args.max_distance is not None:
    crime_nodes_ignored = crime_nodes[crime_nodes["distance"] >= args.max_distance].copy()
    crime_nodes = crime_nodes[crime_nodes["distance"] < args.max_distance]

    with open(crime_processed_path / "crime_nodes_ignored_dataframe.pickle", "wb") as f:
        pickle.dump(crime_nodes_ignored, f)

max_distance_str = str(args.max_distance).replace(".", "m")

with open(crime_processed_path / "crime_nodes_dataframe.pickle", "wb") as f:
    pickle.dump(crime_nodes, f)
