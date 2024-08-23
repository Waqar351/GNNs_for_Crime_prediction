import argparse
import pathlib
import pickle
import copy
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from dataloader.minibatch import MyDataset, get_time_period, calculate_crime_per_node
from utils import compare_columns
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

# output_path = pathlib.Path(f"data/dataloader/dysat/label_column_static_feats/{time_delta}")  ## 1) ORIGINAL code
# output_path = pathlib.Path(f"data/dataloader/dysat/feature_analysis/num_crimes_label_dynamic_feats/{time_delta}")    ## 2) For feature anayslis purpose
# output_path = pathlib.Path(f"data/dataloader/dysat/feature_analysis/num_crimes_Months_exclude_length_2_dynamic_feats/{time_delta}")    ## 3) number of crime per node per time step and crime each month as a dynamic feature.
# output_path = pathlib.Path(f"data/dataloader/dysat/feature_analysis/num_crimes_Months_totCrime_2_1_dynamic_stat_feats/{time_delta}")    ## 4) Include total crime static features number of crime per node per time step and crime each month as a dynamic feature.
# output_path = pathlib.Path(f"data/dataloader/dysat/feature_analysis/num_crimes_Months_totCrime_highway_2_26_dynamic_stat_feats/{time_delta}")    ## 5) Encode Highway and Include total crime static features number of crime per node per time step and crime each month as a dynamic feature.
output_path = pathlib.Path(f"data/dataloader/dysat/feature_analysis/num_crimes_totCrime_highway_1_26_1_dynamic_stat_feats/{time_delta}")    ## 5) Encode Highway and Include total crime static features number of crime per node per time step as a dynamic feature.
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

total_crimes_per_node = calculate_crime_per_node(crime_nodes_df)
edges_dataframe['total_crimes'] = total_crimes_per_node['crime_count']

# Encoding Highway column
edges_dataframe['highway'] = edges_dataframe['highway'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
edges_dataframe = pd.get_dummies(edges_dataframe, columns=['highway'], dtype=int)

highway_columns = ['highway_cycleway',
       'highway_footway', 'highway_living_street',
       'highway_living_street,footway', 'highway_living_street,residential',
       'highway_pedestrian', 'highway_pedestrian,footway',
       'highway_pedestrian,residential', 'highway_primary',
       'highway_primary_link', 'highway_residential',
       'highway_residential,footway', 'highway_secondary',
       'highway_secondary_link', 'highway_service',
       'highway_service,residential', 'highway_steps,pedestrian',
       'highway_steps,pedestrian,footway', 'highway_steps,residential',
       'highway_steps,residential,footway', 'highway_tertiary',
       'highway_tertiary,primary', 'highway_tertiary,residential',
       'highway_tertiary_link', 'highway_trunk', 'highway_trunk_link']
edges_dataframe
dummy_df = edges_dataframe[highway_columns]


# Report results
try:
    # Find duplicate columns and columns with only zeros
    duplicates, zero_columns = compare_columns(dummy_df)

    if duplicates:
        print("Duplicate columns found:")
        for col, duplicates_list in duplicates.items():
            print(f"Column '{col}' is identical to columns: {duplicates_list}")
        raise ValueError("Execution stopped due to duplicate columns.")

    if zero_columns:
        print("\nColumns with only zeros:")
        print(zero_columns)
        raise ValueError("Execution stopped due to columns with only zeros.")

    print("No duplicate columns or columns with only zeros found.")
except ValueError as e:
    print(f"Error: {e}")

# Create empty lists to save Graphs, Adjacency matrices and Features for all time steps
graphs, feats, adjs, feats_analysis = [],[],[],[]
nodes_labels_times = np.array([[None, None, None]])

adj = nx.adjacency_matrix(graph_networkx)
columns_not_used = ['geometry', 'street_segment', 'street_segment_31983', 'distance_tobarueri',	'distance_tointerlagos', 'distance_tomirante']

print(f'\n Creating timeSeries Data according to {time_delta} ...')

for time_step, period in enumerate(time_periods):

    # --------- STATIC Scenario ---------------------------------------------
    graphs.append(graph_networkx)
    adjs.append(adj)
    # feats.append(csr_matrix(edges_dataframe.drop(columns = columns_not_used).values))
    feature_vector = edges_dataframe.drop(columns = columns_not_used)

    # -------------------------------------------------------------------------

    df_Crime_t_step = crime_nodes_df[crime_nodes_df['time'].dt.to_period(args.time_delta) == period]
    df_Crime_t_step = df_Crime_t_step.drop(columns=['time',	'geometry',	'crime_coord','street_segment',	'street_segment_31983','highway'])
    crime_nodes_per_t_step = df_Crime_t_step["index_right"].unique()

    crime_per_node = calculate_crime_per_node(df_Crime_t_step)
    
    feature_vector['node'] = feature_vector.index
    # Merge the DataFrames on the 'node' column
    feature_vector = pd.merge(feature_vector, crime_per_node, on='node', how='left')

    feature_vector['time_step'] = time_step

    # for month in range(1, 13):
    #     feature_vector[f'Month_{month}'] = 0

    # # Extract month from the period
    # crime_month = period.month
    # feature_vector.loc[crime_nodes_per_t_step, f'Month_{crime_month}'] = 1

    # Class label info Must be last column
    feature_vector['label'] = 0
    feature_vector.loc[crime_nodes_per_t_step, 'label']  = 1

    feats_analysis.append(feature_vector)
    feature_vector = feature_vector.drop(columns= ['node', 'time_step'])
    
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
with open(output_path / f"features_analysis__{args.time_delta}.pickle", "wb") as f:
    pickle.dump(feats_analysis, f)

with open(output_path / f"adjacency_all_times__{args.time_delta}.pickle", "wb") as f:
    pickle.dump(adjs, f)


dataset = MyDataset(args, graphs, feats, adjs)

# dataloader = DataLoader(dataset, 
#                         batch_size=args.batch_size, 
#                         shuffle=True, 
#                         num_workers=10, 
#                         collate_fn=MyDataset.collate_fn
#                         )


with open(output_path / f"dataset_dynamic__{args.time_delta}.pickle", "wb") as f:
    pickle.dump(dataset, f)

# with open(output_path / f"dataloader_static__{args.time_delta}.pickle", "wb") as f:
#     pickle.dump(dataloader, f)

print('\n Data created and Saved Successfully!')





