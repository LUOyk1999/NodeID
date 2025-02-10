from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from utils import adjoverlap
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from torch_sparse import SparseTensor
import torch_sparse
from torch_scatter import scatter_add
from typing import Iterable
from typing_extensions import Final
# a vanilla message passing layer 
class PureConv(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x
    

convdict = {
    "gcn":
    GCNConv,
    "gcn_cached":
    lambda indim, outdim: GCNConv(indim, outdim, cached=True),
    "sage":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none":
    None
}

predictor_dict = {}

# Edge dropout
class DropEdge(nn.Module):

    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index: Tensor):
        if self.dp == 0:
            return edge_index
        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp
        return edge_index[:, mask]

# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: Final[bool] # whether to rescale edge weight
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj


# Vanilla MPNN composed of several layers.
class GCN(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False, codebook_size=16, kmeans=1):
        super().__init__()
        
        self.adjdrop = DropAdj(edrop)
        self.kmeans = kmeans
        self.vqs = torch.nn.ModuleList()
        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            if not noinputlin and ("pure" in conv_fn or num_layers==0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
            
        if num_layers == 0 or conv_fn =="none":
            self.jk = False
            return
        
        convfn = convdict[conv_fn]
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            if self.kmeans:
                from vq import VectorQuantize, ResidualVectorQuant
                self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=codebook_size, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
                from vqtorch.nn import VectorQuant, ResidualVectorQuant
                self.vqs.append(ResidualVectorQuant(
                        groups = 3,
                        feature_size=hidden_channels,     # feature dimension corresponding to the vectors
                        num_codes=codebook_size,      # number of codebook vectors
                        beta=0.98,           # (default: 0.9) commitment trade-off
                        kmeans_init=False,    # (default: False) whether to use kmeans++ init
                        norm=None,           # (default: None) normalization for the input vectors
                        cb_norm=None,        # (default: None) normalization for codebook vectors
                        affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                        sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                        replace_freq=20,     # (default: None) frequency to replace dead codes
                        dim=-1,              # (default: -1) dimension to be quantized
                        ))
            for i in range(num_layers-1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
                if self.kmeans:
                    self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=codebook_size, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
                else:
                    self.vqs.append(ResidualVectorQuant(
                            groups = 3,
                            feature_size=hidden_channels,     # feature dimension corresponding to the vectors
                            num_codes=codebook_size,      # number of codebook vectors
                            beta=0.98,           # (default: 0.9) commitment trade-off
                            kmeans_init=False,    # (default: False) whether to use kmeans++ init
                            norm=None,           # (default: None) normalization for the input vectors
                            cb_norm=None,        # (default: None) normalization for codebook vectors
                            affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                            sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                            replace_freq=20,     # (default: None) frequency to replace dead codes
                            dim=-1,              # (default: -1) dimension to be quantized
                            ))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            if self.kmeans:
                from vq import VectorQuantize, ResidualVectorQuant
                self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=codebook_size, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
                from vqtorch.nn import VectorQuant, ResidualVectorQuant
                self.vqs.append(ResidualVectorQuant(
                        groups = 3,
                        feature_size=hidden_channels,     # feature dimension corresponding to the vectors
                        num_codes=codebook_size,      # number of codebook vectors
                        beta=0.98,           # (default: 0.9) commitment trade-off
                        kmeans_init=False,    # (default: False) whether to use kmeans++ init
                        norm=None,           # (default: None) normalization for the input vectors
                        cb_norm=None,        # (default: None) normalization for codebook vectors
                        affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                        sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                        replace_freq=20,     # (default: None) frequency to replace dead codes
                        dim=-1,              # (default: -1) dimension to be quantized
                        ))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                            nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels))
                if self.kmeans:
                    self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=codebook_size, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
                else:
                    self.vqs.append(ResidualVectorQuant(
                            groups = 3,
                            feature_size=hidden_channels,     # feature dimension corresponding to the vectors
                            num_codes=codebook_size,      # number of codebook vectors
                            beta=0.98,           # (default: 0.9) commitment trade-off
                            kmeans_init=False,    # (default: False) whether to use kmeans++ init
                            norm=None,           # (default: None) normalization for the input vectors
                            cb_norm=None,        # (default: None) normalization for codebook vectors
                            affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                            sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                            replace_freq=20,     # (default: None) frequency to replace dead codes
                            dim=-1,              # (default: -1) dimension to be quantized
                            ))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())
        

    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        id_list = []
        quantized_list = []
        total_commit_loss = 0
        # for i, conv in enumerate(self.convs):
        for i, (conv, vq) in enumerate(zip(self.convs, self.vqs)):
            x1 = self.lins[i](conv(x, self.adjdrop(adj_t)))

            if self.kmeans:
                quantized, _, commit_loss, dist, codebook = vq(x1)
                id_list.append(torch.stack(_, dim=1))
                quantized_list.append(quantized)
                total_commit_loss += commit_loss
            else:
                x_, vq_ = vq(x1)
                total_commit_loss += vq_['loss'].mean()
                id_list.append(vq_['q'])

            if self.res and x1.shape[-1] == x.shape[-1]: # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk: # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        
        id_list_concat = torch.cat(id_list, dim=1)
        return x, total_commit_loss, id_list_concat

# CN predictor
class CNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            lnfn(in_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        # print(x)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcns = [spmm_add(cn, x)]
        
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])

predictor_dict = {
    "cn1": CNLinkPredictor,
}