import argparse
import copy
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, load_fixed_splits, eval_rocauc
from dataset import load_nc_dataset
from dataset_large import load_dataset
from logger import Logger
from parse import parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)

warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx


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
    
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, feats=None, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        if args.method == "fast_transgnn" or args.method == "glcn":
            out, _ = model(dataset)
        else:
            out = model(feats)
    # print(id_list_concat.shape)
    # s()
    train_acc = eval_func(dataset.label[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(dataset.label[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(dataset.label[split_idx["test"]], out[split_idx["test"]])
    if args.dataset in (
        "yelp-chi",
        "deezer-europe",
        "twitch-e",
        "fb100",
        "ogbn-proteins",
    ):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(
            out[split_idx["valid"]],
            true_label.squeeze(1)[split_idx["valid"]].to(torch.float),
        )
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx["valid"]], dataset.label.squeeze(1)[split_idx["valid"]]
        )

    return train_acc, valid_acc, test_acc, valid_loss, out

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
if args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
    dataset = load_dataset(args.data_dir, args.dataset)
else:
    dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

dataset_name = args.dataset

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'])
else:
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

_shape = dataset.graph['node_feat'].shape
print(f'features shape={_shape}')

# whether or not to symmetrize
if args.dataset not in {'deezer-europe', 'ogbn-proteins'}:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)
# print(dataset.label)
print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = MLP(num_layers=args.num_layers,
            input_dim=args.num_id,
            hidden_dim=args.hidden_channels,
            output_dim=c,
            dropout_ratio=args.dropout,
            norm_type=args.norm_type).to(device)

# using rocauc as the eval function
if args.dataset in ('deezer-europe', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

eval_func = eval_acc

if args.dataset in ['ogbn-proteins']:
    eval_func = eval_rocauc
logger = Logger(args.runs, args)

model.train()

### Training loop ###
patience = 0

optimizer = torch.optim.Adam(
    model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

def load_out_t(name):
    return torch.from_numpy(np.load(name)["arr_0"])

for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
        
    # np.savez(f"split_idx_{args.dataset}", split_idx['train'])
    # np.load(f"split_idx_{args.dataset}")["arr_0"]

    # if args.dataset in ['ogbn-arxiv']:
    #     print(dataset.load_fixed_splits()['train'].shape)
    #     train = []
    #     for i in dataset.load_fixed_splits()['train']:
    #         # print(dataset.label[i])
    #         if(dataset.label[i][0]<=20):
    #             # print(i, dataset.label[i])
    #             pass
    #         else:
    #             train.append(i)
    #     train = torch.stack(train, dim=0)
    #     train_idx = train
    #     print(train_idx.shape)
    #     # s()
    # elif args.dataset in ['cora', 'citeseer', 'pubmed']:
    #     print(split_idx['train'].shape)
    #     train = []
    #     for i in split_idx['train']:
    #         # print(dataset.label[i])
    #         if(dataset.label[i][0]<1):
    #             # print(i, dataset.label[i])
    #             pass
    #         else:
    #             train.append(i)
    #     train = torch.stack(train, dim=0)
    #     train_idx = train
    #     print(train_idx.shape)
    #     # s()
    # else:
    #     train_idx = split_idx['train'].to(device)
        
    train_idx = split_idx['train'].to(device)
    # print(train_idx)
    model.reset_parameters()

    # GNN_id = load_out_t(f"GNN_ID_{args.dataset}.npz")
    semantic_id = load_out_t(f"semantic_ID_{args.dataset}.npz")
    for i in range(semantic_id.shape[-1]):
        values = semantic_id[..., i].flatten()
        unique_values, counts = torch.unique(values, return_counts=True)
        print(f"location i: {list(zip(unique_values.tolist(), counts.tolist()))}")
    print(semantic_id.shape)
    step = 1
    columns_to_select = [i for i in range(0, semantic_id.size(1), step)]
    # feats = torch.cat([semantic_id,GNN_id],dim=-1).float().to(device)
    feats = semantic_id.float().to(device)[:,columns_to_select]
    # print(feats.shape)
    # feats = dataset.graph['node_feat']
    # model_gnn = SimpleGNNLayer(args.k)
    # feats = model_gnn(feats, dataset.graph['edge_index'])
    # feats = torch.rand(feats.shape).to(device)
    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        emb = None
        out = model(feats)
        if args.dataset in ('deezer-europe', 'ogbn-proteins'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(
                    dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        
        (loss).backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx,
                          eval_func, criterion, args, feats)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            patience = 0
        else:
            patience += 1
            # if patience >= args.patience:
            #     break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)
results = logger.print_statistics()
print(results)
out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

def make_print(method):
    print_str = ''
    if args.rand_split_class:
        print_str += f'label per class:{args.label_num_per_class}, valid:{args.valid_num},test:{args.test_num}\n'
    print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr}\n'
    return print_str


file_name = f'{args.dataset}_{args.method}'
file_name += '.txt'
out_path = os.path.join(out_folder, file_name)
with open(out_path, 'a+') as f:
    print_str = make_print(args.method)
    f.write(print_str)
    f.write(results)
    f.write('\n\n')
