import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.utils import expand_as_pair

from graphmae.utils import create_activation, NormLayer, create_norm


class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum", kmeans=1, num_codes=16
                 ):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        self.vqs = torch.nn.ModuleList()
        self.kmeans = kmeans
        
        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            apply_func = MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            self.layers.append(GINConv(in_dim, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))
            if self.kmeans:
                from graphmae.models.vq import VectorQuantize, ResidualVectorQuant
                print("kmeans")
                self.vqs.append(ResidualVectorQuant(dim=out_dim, codebook_size=num_codes, num_res_layers=3, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
                from vqtorch.nn import VectorQuant, ResidualVectorQuant
                print("vq")
                self.vqs.append(ResidualVectorQuant(
                        groups = 3,
                        feature_size=out_dim,     # feature dimension corresponding to the vectors
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

        else:
            # input projection (no residual)
            self.layers.append(GINConv(
                in_dim, 
                num_hidden, 
                ApplyNodeFunc(MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                init_eps=0,
                learn_eps=learn_eps,
                residual=residual)
                )
            if self.kmeans:
                from graphmae.models.vq import VectorQuantize, ResidualVectorQuant
                print("kmeans")
                print(num_codes)
                self.vqs.append(ResidualVectorQuant(dim=num_hidden, codebook_size=num_codes, num_res_layers=3, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
                from vqtorch.nn import VectorQuant, ResidualVectorQuant
                print("vq")
                self.vqs.append(ResidualVectorQuant(
                        groups = 3,
                        feature_size=num_hidden,     # feature dimension corresponding to the vectors
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
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(GINConv(
                    num_hidden, num_hidden, 
                    ApplyNodeFunc(MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                    init_eps=0,
                    learn_eps=learn_eps,
                    residual=residual)
                )
                if self.kmeans:
                    from graphmae.models.vq import VectorQuantize, ResidualVectorQuant
                    print("kmeans")
                    self.vqs.append(ResidualVectorQuant(dim=num_hidden, codebook_size=num_codes, num_res_layers=3, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
                else:
                    from vqtorch.nn import VectorQuant, ResidualVectorQuant
                    print("vq")
                    self.vqs.append(ResidualVectorQuant(
                            groups = 3,
                            feature_size=num_hidden,     # feature dimension corresponding to the vectors
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
            # output projection
            apply_func = MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)

            self.layers.append(GINConv(num_hidden, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))

            if self.kmeans:
                from graphmae.models.vq import VectorQuantize, ResidualVectorQuant
                print("kmeans")
                self.vqs.append(ResidualVectorQuant(dim=out_dim, codebook_size=num_codes, num_res_layers=3, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
                from vqtorch.nn import VectorQuant, ResidualVectorQuant
                print("vq")
                self.vqs.append(ResidualVectorQuant(
                        groups = 3,
                        feature_size=out_dim,     # feature dimension corresponding to the vectors
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
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        id_list = []
        quantized_list = []
        total_commit_loss = 0
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)
            if self.kmeans:
                quantized, _, commit_loss, dist, codebook = self.vqs[l](h)
                id_list.append(torch.stack(_, dim=1))
                quantized_list.append(quantized)
                total_commit_loss += commit_loss[0]
            else:
                x_, vq_ = self.vqs[l](h.float())
                total_commit_loss += vq_['loss'].mean()
                id_list.append(vq_['q'])
            hidden_list.append(h)
        id_list_concat = torch.cat(id_list, dim=1)

        # output projection
        if return_hidden:
            return self.head(h), hidden_list, total_commit_loss, id_list_concat
        else:
            return self.head(h), total_commit_loss, id_list_concat

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GINConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 apply_func,
                 aggregator_type="sum",
                 init_eps=0,
                 learn_eps=False,
                 residual=False
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.apply_func = apply_func
        
        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
            
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            return rst


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = create_norm(norm)
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)