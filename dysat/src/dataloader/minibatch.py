# from typing import DefaultDict
# from collections import defaultdict
# from torch.functional import Tensor
from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd
import torch_geometric as tg
import scipy.sparse as sp


import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, args, graphs, features, adjs):
        super(MyDataset, self).__init__()
        self.args = args
        self.graphs = graphs
        self.labels = [self._label_creator(feat) for feat in features]
        self.features = [self._preprocess_features(feat) for feat in features]
        self.adjs = [self._normalize_graph_gcn(a)  for a  in adjs]
        self.time_steps = args.time_steps
        # self.max_positive = args.neg_sample_size
        # self.train_nodes = list(self.graphs[self.time_steps-1].nodes()) # all nodes in the graph.
        self.min_t = max(self.time_steps - self.args.window - 1, 0) if args.window > 0 else 0
        # self.degs = self.construct_degs()
        self.pyg_graphs = self._build_pyg_graphs()
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        adj = sp.coo_matrix(adj, dtype=np.float32)          #COO is a fast format for constructing sparse matrices. Once a COO matrix has been constructed, convert to CSR or CSC format for fast arithmetic and matrix vector operations.
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)     # Adjacency Matrix with Self-Loop
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)  #inverse square root of the row sums ##Overall, the power of -0.5 normalization technique helps to improve the stability and effectiveness of graph neural networks by ensuring that nodes contribute equally to the learning process, regardless of their degrees.
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  # the two dot product operations in the last step of the normalization process help to achieve symmetric normalization of the adjacency matrix, thereby ensuring balanced contributions from all nodes in the graph during the message passing process.
        return adj_normalized

    def _preprocess_features(self, features):   # row-wise normalization or row-wise scaling
        """Row-normalize feature matrix and convert to tuple representation"""
        features = np.array(features.todense())     # Convert the feature into dense matrix of [#nodes, Features]
        features = features[:, :-1]               #Excluding last column
        rowsum = np.array(features.sum(1))          # In this particular case the sum is 1 for each row
        r_inv = np.power(rowsum, -1).flatten()      # Take the power of 1 raise to -1 and then flatten it to have single dimenion
        r_inv[np.isinf(r_inv)] = 0.                 # Find infinite indices and replace it to 0.
        r_mat_inv = sp.diags(r_inv)                 # Create a sparse diagonal matrix with the inverse row sums
        features = r_mat_inv.dot(features)          # Row-normalize the feature matrix ##each row of the feature matrix is element-wise multiplied by the corresponding reciprocal of the row sum
        
        return features
    
    def _label_creator(self, feat):                # Extracting label information from feature vector
        featur = np.array(feat.todense())
        label = featur[:, -1]
        return label

    # def construct_degs(self):
    #     """ Compute node degrees in each graph snapshot."""
    #     # different from the original implementation
    #     # degree is counted using multi graph
    #     degs = []
    #     for i in range(self.min_t, self.time_steps):
    #         print('timestep in minibatch', i)
    #         G = self.graphs[i]
    #         deg = []
    #         for nodeid in G.nodes():
    #             deg.append(G.degree(nodeid))
    #         degs.append(deg)
    #     return degs

    def _build_pyg_graphs(self):        # It receives all features and adjacency matrix togather
        pyg_graphs = []
        for feat, adj, labl in zip(self.features, self.adjs, self.labels):
            x = torch.Tensor(feat)
            y = torch.Tensor(labl)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)  # This function converts a Scipy sparse matrix representation of a graph's adjacency matrix into PyTorch Geometric's internal representation, which consists of edge index and edge weight tensors.
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y = y)
            pyg_graphs.append(data)
        return pyg_graphs

    def __len__(self):
        return self.time_steps

    def __getitem__(self, index):
        # graph = self.graphs[index]
        return self.data_items[index]
    
    def __createitems__(self):
        self.data_items = {}
        # for node in list(self.graphs[self.time_steps-1].nodes()):
        for t in range(self.min_t, self.time_steps):
            feed_dict = {}
            feed_dict["graphs"] = self.pyg_graphs
        
            self.data_items[t] = feed_dict

    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        # for key in ["node_1", "node_2", "node_2_neg"]:
        #     data_list = []
        #     for sample in samples:
        #         data_list.append(sample[key])
        #     concate = []
        #     for t in range(len(data_list[0])):
        #         concate.append(torch.cat([data[t] for data in data_list]))
        #     batch_dict[key] = concate
        batch_dict["graphs"] = samples[0]["graphs"]
        return batch_dict

def get_time_period(args_time_delta):
    """
    Returns the time period corresponding to the given time delta.

    Parameters:
    args_time_delta (str): A single character representing the time delta.
                           "D" for Days, "M" for Months, "Y" for Years.

    Returns:
    str: The time period as a string.
    """
    time_deltas = {
        "D": "Days",
        "M": "Months",
        "Y": "Years"
    }
    
    # Check if the input is valid
    if args_time_delta in time_deltas:
        return time_deltas[args_time_delta]
    else:
        return "Invalid time delta. Please use 'D' for Days, 'M' for Months, or 'Y' for Years."
    

def select_time_steps(time_delta):
    """
    Choose the default number of time steps based on the time delta.

    Parameters:
    time_delta (str): The time delta which can be "D", "M", or "Y".

    Returns:
    int: The calculated number of time steps.
    """
    if time_delta == "D":
        return 31  # Default for days
    elif time_delta == "M":
        return 3  # Default for months
    elif time_delta == "Y":
        return 0   # Default for years
    else:
        raise ValueError("Invalid time delta. Please use 'D' for Days, 'M' for Months, or 'Y' for Years.")
    
def calculate_crime_per_node(dataframe):
        # Calculate the number of crime occurrences for each node
    crime_counts = dataframe['index_right'].value_counts()
    crime_counts_df = pd.DataFrame({'node': crime_counts.index, 'crime_count': crime_counts.values})
    crime_counts_df = crime_counts_df.sort_values(by='node')

    full_node_range = pd.DataFrame({'node': range(4121)})
    # Initialize the 'crime_count' column to 0
    full_node_range['crime_count'] = 0
    # Merge with the existing DataFrame to fill in crime counts
    merged_df = pd.merge(full_node_range, crime_counts_df, 
                     on='node', how='left')

    # Fill NaN values in 'crime_count' with 0 for missing nodes
    merged_df['crime_count'] = merged_df['crime_count_y'].fillna(0)
    merged_df = merged_df.drop(columns=['crime_count_x',	'crime_count_y'])

    return merged_df