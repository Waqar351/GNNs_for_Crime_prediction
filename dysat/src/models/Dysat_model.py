# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2024/08/27 21:10:00
@Author  :   Waqar Hassan 
@Contact :   waqar_comsat@yahoo.com
USP, Sao Paulo, Brazil
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
import numpy as np

from models.layers import StructuralAttentionLayer, TemporalAttentionLayer
import pdb

class DySAT(nn.Module):
    def __init__(self, args, num_features, time_length, class_weight): #(143, 16)
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.args = args
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features
        self.structural_head_config = list(map(int, args.structural_head_config.split(","))) # (16,8,8)
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(","))) #(128)
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(","))) # (16)
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(","))) #(128)
        self.spatial_drop = args.spatial_drop # (0.1)
        self.temporal_drop = args.temporal_drop # (0.5)
        self.structural_attn, self.temporal_attn, self.fc_binary = self.build_model()
        self.bceloss = BCEWithLogitsLoss(pos_weight = class_weight)

    def forward(self, graphs):

        # Structural Attention forward
        structural_out = []
        for t in range(0, self.num_time_steps - 1):
            structural_out.append(self.structural_attn(graphs[t]))  # Structural output for each time step
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]  RESHAPING from [Ni, F] --> [Ni, 1, F]
 
        structural_outputs = torch.cat(structural_outputs, dim=1) # [N, T, F]
        ### Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs)
        temporal_out = temporal_out[:, -1, :]

        binary_output = self.fc_binary(temporal_out)  # [N, 1] for each node
        binary_output = torch.sigmoid(binary_output).squeeze()
        
        return binary_output

    def build_model(self):
        input_dim = self.num_features

        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.args.residual)
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]
        
        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]
        
        # Adding classification layer
        fc_binary = nn.Linear(in_features=input_dim, out_features=1)
        

        return structural_attention_layers, temporal_attention_layers, fc_binary

    def get_loss(self, feed_dict):
        graphs = []
        predict_probs = []
        # graphs = feed_dict['graphs']
        graphs = feed_dict
        # run gnn
        predict_probs = self.forward(graphs) # [N, T, F]
        self.graph_loss = 0
        actual_labels = graphs[-1].y
        self.graph_loss = self.bceloss(predict_probs, actual_labels)
        return self.graph_loss, predict_probs

            




