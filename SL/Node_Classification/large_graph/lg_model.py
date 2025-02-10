import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3,
            in_dropout=0.15, dropout=0.5, heads=1,
            beta=0.9, pre_ln=False, post_bn=True, local_attn=False, kmeans=1, num_codes=16):
        super(GAT, self).__init__()

        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.post_bn = post_bn

        ## Two initialization strategies on beta
        self.beta = beta
        #self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)
        self.vqs = torch.nn.ModuleList()
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.post_bn:
            self.post_bns = torch.nn.ModuleList()

        ## first layer
        self.h_lins.append(torch.nn.Linear(in_channels, heads*hidden_channels))
        if local_attn:
            self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                concat=True, add_self_loops=False, bias=False))
        else:
            self.local_convs.append(SAGEConv(in_channels, heads*hidden_channels,
                cached=False, normalize=True))

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
            
        self.lins.append(torch.nn.Linear(in_channels, heads*hidden_channels))
        self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        if self.pre_ln:
            self.pre_lns.append(torch.nn.LayerNorm(in_channels))
        if self.post_bn:
            self.post_bns.append(torch.nn.BatchNorm1d(heads*hidden_channels))

        ## following layers
        for _ in range(local_layers-1):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if local_attn:
                self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            else:
                self.local_convs.append(SAGEConv(heads*hidden_channels, heads*hidden_channels,
                    cached=False, normalize=True))

            if self.kmeans:
                print("kmeans")
                self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=num_codes, num_res_layers=3, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
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
            
            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.post_bn:
                self.post_bns.append(torch.nn.BatchNorm1d(heads*hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        self.pred_local = torch.nn.Linear(heads*hidden_channels, out_channels)
        self.linear_gnn = torch.nn.Linear(heads*hidden_channels, local_layers*3)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.post_bn:
            for p_bn in self.post_bns:
                p_bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.pred_local.reset_parameters()
        #torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)

        id_list = []
        quantized_list = []
        total_commit_loss = 0
        
        ## equivariant local attention
        x_local = 0
        for i, (local_conv, vq) in enumerate(zip(self.local_convs, self.vqs)):
            if self.pre_ln:
                x = self.pre_lns[i](x)

            x = local_conv(x, edge_index) + self.lins[i](x)
            if self.post_bn:
                x = self.post_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x_local = x_local + x

            if self.kmeans:
                quantized, _, commit_loss, dist, codebook = vq(x)
                id_list.append(torch.stack(_, dim=1))
                quantized_list.append(quantized)
                total_commit_loss += commit_loss
            else:
                x_, vq_ = vq(x)
                total_commit_loss += vq_['loss'].mean()
                id_list.append(vq_['q'])

        id_list_concat = torch.cat(id_list, dim=1)
        
        gnn_id = self.linear_gnn(x_local)

        x = self.pred_local(x_local)

        return x, total_commit_loss, id_list_concat, gnn_id
