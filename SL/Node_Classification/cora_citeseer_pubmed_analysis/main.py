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
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits, eval_rocauc
from dataset import load_nc_dataset
from dataset_large import load_dataset
from logger import Logger
from parse import parse_method, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)

warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args.method, args, c, d, device)

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

import pickle

for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
        
    # if args.dataset in ['ogbn-arxiv']:
    #     print(dataset.load_fixed_splits()['train'].shape)
    #     train = []
    #     for i in dataset.load_fixed_splits()['train']:
    #         # print(dataset.label[i])
    #         if(dataset.label[i][0]<20):
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
    # print(split_idx)
    # with open(f'{args.dataset}_split.pickle', 'wb') as f:
    #     pickle.dump(split_idx, f)

    train_idx = split_idx['train'].to(device)
    # print(train_idx.shape)
    model.reset_parameters()

    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        emb = None
        out, total_commit_loss, id_list_concat, _ = model(dataset)
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
        
        (loss+total_commit_loss).backward()
        optimizer.step()
        # if epoch % 10 == 0:
        if args.dataset in ('ogbn-arxiv'):
            if epoch % 10 == 0:
                result = evaluate(model, dataset, split_idx,
                            eval_func, criterion, args)
                logger.add_result(run, result[:-1])

                if result[1] > best_val:
                    best_val = result[1]
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        break
                
                print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * result[0]:.2f}%, '
                    f'Valid: {100 * result[1]:.2f}%, '
                    f'Test: {100 * result[2]:.2f}%')
        else:
            result = evaluate(model, dataset, split_idx,
                            eval_func, criterion, args)
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break
            
    print("training is over!", loss, total_commit_loss, train_idx[-10:])
    
