import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from torch_scatter import scatter

from lg_parse import parse_method, parser_add_main_args
import sys
sys.path.append("../")
from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits


import warnings
warnings.filterwarnings('ignore')



@torch.no_grad()
def evaluate_large(model, feats, dataset, split_idx, eval_func, criterion, args, device="cpu", result=None):
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device(device))
    dataset.label = dataset.label.to(torch.device(device))
    edge_index, x = dataset.graph['edge_index'].to(torch.device(device)), dataset.graph['node_feat'].to(torch.device(device))
    out = model(feats.to(torch.device(device)))

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out



from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class SimpleGNNLayer(MessagePassing):
    def __init__(self, k=10):
        super(SimpleGNNLayer, self).__init__(aggr='add')
        self.k = k
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        norm = deg.pow(-0.5).unsqueeze(1)
        for _ in range(self.k):
            x = x * norm
            x = self.propagate(edge_index, x=x)
            x = x * norm

        return x

    def message(self, x_j):
        return x_j

import torch.nn.functional as F
import torch.nn as nn
class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()
        for bn in self.norms:
            bn.reset_parameters()
            
    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            # print(h.shape)
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    
# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

# get the splits for all runs
if args.dataset in ('ogbn-arxiv', 'ogbn-products'):
    split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")


### Load method ###
model = MLP(num_layers=args.num_layers,
            input_dim=args.num_id,
            hidden_dim=args.hidden_channels,
            output_dim=c,
            dropout_ratio=args.dropout,
            norm_type=args.norm_type).to(device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']

if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    else:
        true_label = dataset.label
else:
    true_label = dataset.label

def load_out_t(name):
    return torch.from_numpy(np.load(name)["arr_0"])

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'pokec']:
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    print(train_mask[-10:])
    model.reset_parameters()
    
    semantic_id = load_out_t(f"semantic_ID_{args.dataset}.npz")
    for i in range(semantic_id.shape[-1]):
        values = semantic_id[..., i].flatten()
        unique_values, counts = torch.unique(values, return_counts=True)
        print(f"location i: {list(zip(unique_values.tolist(), counts.tolist()))}")
    print(semantic_id.shape)
    feats = torch.tensor(semantic_id).float()
    # feats = torch.rand(feats.shape).to(device)

    
    
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    num_batch = n // args.batch_size + (n%args.batch_size>0)
    for epoch in range(args.epochs):
        model.to(device)
        model.train()

        idx = torch.randperm(n)
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            train_mask_i = train_mask[idx_i]
            x_i = feats[idx_i].to(device)

            y_i = true_label[idx_i].to(device)
            optimizer.zero_grad()
            out_i = model(x_i)
            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i].to(torch.float))

            else:
                out_i = F.log_softmax(out_i, dim=1)
                loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])
            (loss).backward()
            optimizer.step()

        if epoch % args.eval_step == 0:
            
            result = evaluate_large(model, feats, dataset, split_idx, eval_func, criterion, args, device="cpu")
            logger.add_result(run, result[:-1])

            print_str = f'Epoch: {epoch:02d}, ' + \
                        f'Loss: {loss:.4f}, ' + \
                        f'Train: {100 * result[0]:.2f}%, ' + \
                        f'Valid: {100 * result[1]:.2f}%, ' + \
                        f'Test: {100 * result[2]:.2f}%'
            print(print_str)
    logger.print_statistics(run)

logger.print_statistics()