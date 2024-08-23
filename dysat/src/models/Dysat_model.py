# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
import numpy as np

from models.layers import StructuralAttentionLayer, TemporalAttentionLayer
# from utils.utilities import fixed_unigram_candidate_sampler
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
        # self.nodes_labels_times = node_labels.astype(np.int32)

        self.structural_head_config = list(map(int, args.structural_head_config.split(","))) # (16,8,8)
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(","))) #(128)
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(","))) # (16)
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(","))) #(128)
        self.spatial_drop = args.spatial_drop # (0.1)
        self.temporal_drop = args.temporal_drop # (0.5)

        self.structural_attn, self.temporal_attn, self.fc_binary = self.build_model()

        # self.bceloss = BCEWithLogitsLoss(weight = torch.Tensor([0.01, 0.99]))
        self.bceloss = BCEWithLogitsLoss(pos_weight = class_weight)
        # self.bceloss = BCEWithLogitsLoss(reduction = 'none')

    def forward(self, graphs):

        # Structural Attention forward
        structural_out = []
        for t in range(0, self.num_time_steps - 1):
            structural_out.append(self.structural_attn(graphs[t]))  # Structural output for each time step
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]  RESHAPING from [Ni, F] --> [Ni, 1, F]

        # #### padding outputs along with Ni
        # #### to pad the embeddings along the node dimension to make them all have the same number of nodes.
        # maximum_node_num = structural_outputs[-1].shape[0] 
        # out_dim = structural_outputs[-1].shape[-1]
        # structural_outputs_padded = []
        # for out in structural_outputs:
        #     zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)  #Calculate the amount of zero padding needed for each embedding tensor
        #     padded = torch.cat((out, zero_padding), dim=0)      #Concatenate the original embedding tensor with the zero padding along the node dimension
        #     structural_outputs_padded.append(padded)
        # structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]
        
        structural_outputs = torch.cat(structural_outputs, dim=1) # [N, T, F]
        ### Temporal Attention forward
        # temporal_out = self.temporal_attn(structural_outputs_padded)  # USING Padded output
        temporal_out = self.temporal_attn(structural_outputs)
        temporal_out = temporal_out[:, -1, :]

        # Apply binary classification layer  #last ts
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
        # node_1, node_2, node_2_negative, graphs = feed_dict.values()
        graphs = []
        predict_probs = []
        # graphs = feed_dict['graphs']
        graphs = feed_dict
        # run gnn
        predict_probs = self.forward(graphs) # [N, T, F]
        self.graph_loss = 0
        # actual_labels = torch.Tensor(self.nodes_labels_times[np.where(self.nodes_labels_times[:, 2] == last_index_graph)[0], 1])
        actual_labels = graphs[-1].y
        self.graph_loss = self.bceloss(predict_probs, actual_labels)
        # for t in range(0, self.num_time_steps):
            
        #     predict_probs_per_step = predict_probs[:,t,:].squeeze()
        #     graphloss = self.bceloss(predict_probs_per_step, actual_labels)
        #     self.graph_loss += graphloss
        ##############################################

        # weight_positive = 10
        # weight_negative = 1
        # self.graph_loss[actual_labels == 1] *= weight_positive
        # self.graph_loss[actual_labels == 0] *= weight_negative
        # self.graph_loss = self.graph_loss.mean()
        ###############################################
        # for t in range(0, self.num_time_steps - 1):
        #     labels_per_step = torch.Tensor(self.nodes_labels_times[np.where(self.nodes_labels_times[:, 2] == t)[0], 1])
        #     predict_probs_per_step = predict_probs[:,t,:].squeeze()
        #     # bceloss = BCEWithLogitsLoss(weight = torch.tensor([0.01, 0.99]))
        #     graphloss = self.bceloss(predict_probs_per_step, labels_per_step)
        #     self.graph_loss += graphloss
            # print(graphloss)
            # print(labels_time_step_1)

            # emb_t = final_emb[:, t, :].squeeze() #[N, F]  Embeddings for each time step
            # source_node_emb = emb_t[node_1[t]]
            # tart_node_pos_emb = emb_t[node_2[t]]
            # tart_node_neg_emb = emb_t[node_2_negative[t]]
            # pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            # neg_score = -torch.sum(source_node_emb[:, None, :]*tart_node_neg_emb, dim=2).flatten()
            # pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            # neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            # graphloss = pos_loss + self.args.neg_weight*neg_loss  # neg_weight is 1
            # self.graph_loss += graphloss
        # print('loss in model part', self.graph_loss)
        return self.graph_loss, predict_probs

            




