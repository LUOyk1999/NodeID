import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from parse import parse_method, parser_add_main_args


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    # print(dataset.label)
    
    train_acc = eval_func(dataset.label[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(dataset.label[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(dataset.label[split_idx["test"]], out[split_idx["test"]])
    
    if args.dataset in ('questions'):
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


### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
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
dataset.label = dataset.label.to(device)

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)


### Load method ###
model = MLP(num_layers=args.num_layers,
            input_dim=args.num_id,
            hidden_dim=args.hidden_channels,
            output_dim=c,
            dropout_ratio=args.dropout,
            norm_type=args.norm_type).to(device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

def load_out_t(name):
    return torch.from_numpy(np.load(name)["arr_0"])

### Training loop ###
for run in range(args.runs):
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    print(train_idx[-10:])
    # GNN_id = load_out_t(f"GNN_ID_{args.dataset}.npz")
    semantic_id = load_out_t(f"semantic_ID_{args.dataset}.npz")
    for i in range(semantic_id.shape[-1]):
        values = semantic_id[..., i].flatten()
        unique_values, counts = torch.unique(values, return_counts=True)
        print(f"location {i}: {list(zip(unique_values.tolist(), counts.tolist()))}")
    print(semantic_id.shape)
    # feats = torch.cat([semantic_id,GNN_id],dim=-1).float().to(device)
    feats = semantic_id.float().to(device)
    # feats = torch.rand(feats.shape).to(device)    

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')

    for epoch in range(args.epochs):
        
        model.train()
        optimizer.zero_grad()

        out = model(feats)
        if args.dataset in ('questions'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
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

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args, feats)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()


