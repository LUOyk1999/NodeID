import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True, num_codes=32, kmeans=0):
        super(GCN, self).__init__()
        self.vqs = torch.nn.ModuleList()
        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))
        self.kmeans = kmeans
        if self.kmeans:
            from vq import VectorQuantize, ResidualVectorQuant
            print("kmeans")
            self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=num_codes, num_res_layers=3, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
        else:
            from vqtorch.nn import VectorQuant, ResidualVectorQuant
            print("vq")
            self.vqs.append(ResidualVectorQuant(
                    groups = 3,
                    feature_size=hidden_channels,     # feature dimension corresponding to the vectors
                    num_codes=num_codes,      # number of codebook vectors
                    beta=0.98,           # (default: 0.9) commitment trade-off
                    kmeans_init=False,    # (default: False) whether to use kmeans++ init
                    norm=None,           # (default: None) normalization for the input vectors
                    cb_norm=None,        # (default: None) normalization for codebook vectors
                    affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                    sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                    replace_freq=20,     # (default: None) frequency to replace dead codes
                    dim=-1,              # (default: -1) dimension to be quantized
                    ))
        
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            if self.kmeans:
                self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=num_codes, num_res_layers=3, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
                self.vqs.append(ResidualVectorQuant(
                    groups = 3,
                    feature_size=hidden_channels,     # feature dimension corresponding to the vectors
                    num_codes=num_codes,      # number of codebook vectors
                    beta=0.98,           # (default: 0.9) commitment trade-off
                    kmeans_init=False,    # (default: False) whether to use kmeans++ init
                    norm=None,           # (default: None) normalization for the input vectors
                    cb_norm=None,        # (default: None) normalization for codebook vectors
                    affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                    sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                    replace_freq=20,     # (default: None) frequency to replace dead codes
                    dim=-1,              # (default: -1) dimension to be quantized
                    ))
        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
        if self.kmeans:
            self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=num_codes, num_res_layers=3, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
        else:
            self.vqs.append(ResidualVectorQuant(
                    groups = 3,
                    feature_size=hidden_channels,     # feature dimension corresponding to the vectors
                    num_codes=num_codes,      # number of codebook vectors
                    beta=0.98,           # (default: 0.9) commitment trade-off
                    kmeans_init=False,    # (default: False) whether to use kmeans++ init
                    norm=None,           # (default: None) normalization for the input vectors
                    cb_norm=None,        # (default: None) normalization for codebook vectors
                    affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                    sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                    replace_freq=20,     # (default: None) frequency to replace dead codes
                    dim=-1,              # (default: -1) dimension to be quantized
                    ))
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.linear_gnn = nn.Linear(hidden_channels, num_layers*3)
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        # self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers-1,))))
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index=data.graph['edge_index']
        edge_weight=data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        id_list = []
        quantized_list = []
        total_commit_loss = 0
        # x_all = []
        layer_ = []
        edge_index = SparseTensor(row=edge_index[0], col=edge_index[1])
        for i, (conv, vq) in enumerate(zip(self.convs[:-1], self.vqs[:-1])):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x,edge_index,edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # x_all.append(x)
            if self.kmeans:
                quantized, _, commit_loss, dist, codebook = vq(x)
                id_list.append(torch.stack(_, dim=1))
                quantized_list.append(quantized)
                total_commit_loss += commit_loss
            else:
                x_, vq_ = vq(x)
                total_commit_loss += vq_['loss'].mean()
                id_list.append(vq_['q'])

        # jkx = torch.stack(x_all, dim=0)
        # sftmax = self.jkparams.reshape(-1, 1, 1)
        # x = torch.sum(jkx*sftmax, dim=0)
        
        x = self.convs[-1](x, edge_index)
        if self.kmeans:
            quantized, _, commit_loss, dist, codebook = self.vqs[-1](x)
            id_list.append(torch.stack(_, dim=1))
            quantized_list.append(quantized)
            total_commit_loss += commit_loss
        else:
            x_, vq_ = self.vqs[-1](x)
            total_commit_loss += vq_['loss'].mean()
            id_list.append(vq_['q'])
            
        id_list_concat = torch.cat(id_list, dim=1)
        
        return self.linear(x), total_commit_loss, id_list_concat, self.linear_gnn(x)
