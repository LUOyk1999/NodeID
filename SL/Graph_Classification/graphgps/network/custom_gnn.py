import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcn_conv_layer import GCNConvLayer
from torch_geometric.nn import global_add_pool

@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        self.vqs = torch.nn.ModuleList()
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."
        self.kmeans = 1
        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
            if self.kmeans:
                from graphgps.network.vq import VectorQuantize, ResidualVectorQuant
                print("kmeans")
                self.vqs.append(ResidualVectorQuant(dim=cfg.gnn.dim_inner, codebook_size=16, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
                from vqtorch.nn import VectorQuant, ResidualVectorQuant
                print("vq")
                self.vqs.append(ResidualVectorQuant(
                        groups = 3,
                        feature_size=cfg.gnn.dim_inner,     # feature dimension corresponding to the vectors
                        num_codes=16,      # number of codebook vectors
                        beta=0.98,           # (default: 0.9) commitment trade-off
                        kmeans_init=False,    # (default: False) whether to use kmeans++ init
                        norm=None,           # (default: None) normalization for the input vectors
                        cb_norm=None,        # (default: None) normalization for codebook vectors
                        affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                        sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                        replace_freq=20,     # (default: None) frequency to replace dead codes
                        dim=-1,              # (default: -1) dimension to be quantized
                        ))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'gcnconv':
            return GCNConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        
        batch = self.encoder(batch)
        id_list = []
        quantized_list = []
        total_commit_loss = 0
        for (conv, vq) in zip(self.gnn_layers, self.vqs):
            batch = conv(batch)
            if self.kmeans:
                quantized, _, commit_loss, dist, codebook = vq(batch.x)
                id_list.append(torch.stack(_, dim=1))
                quantized_list.append(quantized)
                total_commit_loss += commit_loss
            else:
                x_, vq_ = vq(batch.x)
                total_commit_loss += vq_['loss'].mean()
                id_list.append(vq_['q'])
        id_list_concat = torch.cat(id_list, dim=1)
        graph_id = global_add_pool(id_list_concat, batch.batch)
        batch = self.post_mp(batch)
        # print(graph_id.shape)
        return batch, total_commit_loss, graph_id
