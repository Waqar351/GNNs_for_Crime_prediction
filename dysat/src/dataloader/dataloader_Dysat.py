import argparse
import pathlib
import pickle
import copy
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from minibatch import MyDataset, get_time_period
# from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()


parser.add_argument(
    "--process_id",
    "-id",
    default=None,
    type=str,
    help="Id to idenfity processed data.",
)

parser.add_argument(
    "--time_delta",
    "-td",
    default="D",
    choices=["D", "M", "Y"],
    type=str,
    help="Time period to consider when grouping crimes in (Y)ear, (D)ay or (M)onth. Default is (M)onth.",
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite data.",
)

parser.add_argument('--time_steps', type=int, nargs='?', default=31,
                    help="total time steps used for train, eval and test")

# parser.add_argument('--batch_size', type=int, nargs='?', default=5,  #512 default
#                     help='Batch size (# nodes)')

parser.add_argument('--window', type=int, nargs='?', default=-1,
                    help='Window for temporal attention (default : -1 => full)')

args = parser.parse_args()

folder_name = args.process_id

print('\n Loading Data ...')

processed_data_path = pathlib.Path(f"data/processed/sp/{folder_name}")

time_delta = get_time_period(args.time_delta)

output_path = pathlib.Path(f"data/dataloader/dysat/label_column_static_feats/{time_delta}")
output_path.mkdir(exist_ok=True, parents=True)

assert (
    processed_data_path.is_dir()
), "Invalid processed data. Did you run the correct processing script using the flag --proces_id?"

# Reading data
with open(processed_data_path / "crime_nodes_dataframe.pickle", "rb") as f:
    crime_df = pickle.load(f)

with open(processed_data_path / "gdf_edges.pickle", "rb") as f:
    edges_df = pickle.load(f)

with open(processed_data_path / "graph-streets_as_nodes.pickle", "rb") as f:
    graph_nx = pickle.load(f)

print('\n Loading Complete!')

edges_dataframe = copy.deepcopy(edges_df)
graph_networkx = copy.deepcopy(graph_nx)
crime_nodes_df = copy.deepcopy(crime_df)


# Extracting those nodes which never have any crime occurences in the data "Crime_nodes_dataframe.pickle" 

if not (output_path / "nodes_never_crime.pickle").is_file() or args.overwrite:
    Total_nodes = set(list(graph_networkx.nodes()))
    crime_nodes = set(crime_nodes_df['index_right'])
    nodes_never_crime = list(Total_nodes - crime_nodes)

    with open(output_path / "nodes_never_crime.pickle", "wb") as f:
        pickle.dump(nodes_never_crime, f)
else:
    with open(output_path / "nodes_never_crime.pickle", "rb") as f:
        nodes_never_crime = pickle.load(f)

###_________ STATIC Graph Scenario_______________  ######

# Extracting & creating features, graph & adjacency matrices

total_nodes = len(graph_networkx.nodes())
time_periods = np.sort(crime_nodes_df["time"].dt.to_period(args.time_delta).unique())

# Create empty lists to save Graphs, Adjacency matrices and Features for all time steps
graphs, feats, adjs = [],[],[]
nodes_labels_times = np.array([[None, None, None]])

adj = nx.adjacency_matrix(graph_networkx)
columns_not_used = ['geometry', 'highway', 'street_segment', 'street_segment_31983', 'distance_tobarueri',	'distance_tointerlagos', 'distance_tomirante']

print(f'\n Creating timeSeries Data according to {time_delta} ...')

for time_step, period in enumerate(time_periods):

    # --------- STATIC Scenario ---------------------------------------------
    graphs.append(graph_networkx)
    adjs.append(adj)
    feature_vector = edges_dataframe.drop(columns = columns_not_used)

    # -------------------------------------------------------------------------

    df_Crime_t_step = crime_nodes_df[crime_nodes_df['time'].dt.to_period(args.time_delta) == period]
    df_Crime_t_step = df_Crime_t_step.drop(columns=['time',	'geometry',	'crime_coord','street_segment',	'street_segment_31983','highway'])
    crime_nodes_per_t_step = df_Crime_t_step["index_right"].unique()


    feature_vector['label'] = 0
    feature_vector.loc[crime_nodes_per_t_step, 'label']  = 1
    feats.append(csr_matrix(feature_vector.values))


    nodes_labels = np.zeros((total_nodes, 3), dtype=int)
    nodes_labels[:, 0] = np.arange(total_nodes)
    nodes_labels[:, 2] = time_step
    
    nodes_labels[crime_nodes_per_t_step, 1] += 1

    nodes_labels_times = np.concatenate([nodes_labels_times, nodes_labels])
    
nodes_labels_times = nodes_labels_times[1:]


with open(output_path / f"nodes_labels_times__{args.time_delta}.pickle", "wb") as f:
    pickle.dump(nodes_labels_times, f)

with open(output_path / f"graphs_all_times__{args.time_delta}.pickle", "wb") as f:
    pickle.dump(graphs, f)

with open(output_path / f"features_all_times__{args.time_delta}.pickle", "wb") as f:
    pickle.dump(feats, f)

with open(output_path / f"adjacency_all_times__{args.time_delta}.pickle", "wb") as f:
    pickle.dump(adjs, f)


dataset = MyDataset(args, graphs, feats, adjs)


with open(output_path / f"dataset_static__{args.time_delta}.pickle", "wb") as f:
    pickle.dump(dataset, f)

print('\n Data created and Saved Successfully!')