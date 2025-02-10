"""
Training on large dataset using neighbor sampling.
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
# from data_utils import load_fixed_splits

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.utils import add_self_loops, to_undirected

import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import stats
import graphgps.metrics_ogb as metrics_ogb
from graphgps.metric_wrapper import MetricWrapper
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, \
    confusion_matrix
from sklearn.metrics import r2_score
def eval_spearmanr(y_true, y_pred):
    """Compute Spearman Rho averaged across tasks.
    """
    res_list = []

    if y_true.ndim == 1:
        res_list.append(stats.spearmanr(y_true, y_pred)[0])
    else:
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = ~np.isnan(y_true[:, i])
            res_list.append(stats.spearmanr(y_true[is_labeled, i],
                                            y_pred[is_labeled, i])[0])

    return {'spearmanr': sum(res_list) / len(res_list)}

def regression(true, pred):
        reformat = lambda x: round(float(x), 5)
        return {
            'mae': reformat(mean_absolute_error(true, pred)),
            'r2': reformat(r2_score(true, pred, multioutput='uniform_average')),
            'mse': reformat(mean_squared_error(true, pred)),
            'rmse': reformat(mean_squared_error(true, pred, squared=False)),
        }
    
def l1_losses(pred, true):
    l1_loss = nn.L1Loss()
    loss = l1_loss(pred, true)
    return loss, pred

        
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
    
def index2mask(idx, size: int):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def train(model, feats, y, loss_func, optimizer):
    model.train()
    output = model(feats)
    labels = y
    print(output.shape, labels.shape)
    loss, _ = l1_losses(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, feats, y):
    model = model.eval()
    output = model(feats)
    output = output
    labels = y
    loss, pred_score = l1_losses(output, labels)
    # print(pred_score)
    # print(classification_multilabel(labels,pred_score))
    return regression(labels.detach().to('cpu', non_blocking=True),pred_score.detach().to('cpu', non_blocking=True))['mae'], loss

def make_print(args):
    method = args.method
    print_str = f"batch size: {args.batch_size} "
    if args.use_pretrained:
        print_str += f"use pretrained: {args.model_dir} "
    if method == "gcn":
        print_str += f"method: {args.method} layers:{args.num_layers} hidden: {args.hidden_channels} lr:{args.lr} decay:{args.weight_decay}\n"
    elif method == "ours":
        print_str += (
            f"method: {args.method} hidden:{args.hidden_channels} lr:{args.lr} \n"
        )

    return print_str

import numpy as np
def load_out_t(name):
    return torch.from_numpy(np.load(name)["arr_0"])

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--num_id', type=int, default=18)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--norm_type', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    # --- load data --- #
    print("Start loading dataset")

    output_channels = 11
    dataset = "peptides-structural"
        
    device = torch.device(f"cuda:{args.device}")
    print("Finish loading dataset")

    semantic_id_test = load_out_t(f"semantic_ID_{dataset}_test_id.npz")
    semantic_id_valid = load_out_t(f"semantic_ID_{dataset}_valid_id.npz")
    semantic_id_train = load_out_t(f"semantic_ID_{dataset}_train_id.npz")

    
    # for i in range(semantic_id_train.shape[-1]):
    #     values = semantic_id_train[..., i].flatten()
    #     unique_values, counts = torch.unique(values, return_counts=True)
    #     print(f"location i: {list(zip(unique_values.tolist(), counts.tolist()))}")
    # print(semantic_id_train.shape)
    
    
    semantic_y_test = load_out_t(f"semantic_ID_{dataset}_test_y.npz")
    semantic_y_valid = load_out_t(f"semantic_ID_{dataset}_valid_y.npz")
    semantic_y_train = load_out_t(f"semantic_ID_{dataset}_train_y.npz")
    
    
    feats_train = torch.tensor(semantic_id_train).float().to(device)
    feats_valid = torch.tensor(semantic_id_valid).float().to(device)
    feats_test = torch.tensor(semantic_id_test).float().to(device)
    y_train = semantic_y_train.to(device)
    y_valid = semantic_y_valid.to(device)
    y_test = semantic_y_test.to(device)

    print(feats_train.shape, feats_valid.shape, feats_test.shape)

    # feats_train = model_gnn(feats_train, dataset.graph['edge_index']).to(device)

    
    # --- init model --- #
    model = MLP(num_layers=args.num_layers,
            input_dim=args.num_id,
            hidden_dim=args.hidden_channels,
            output_dim=output_channels,
            dropout_ratio=args.dropout,
            norm_type="batch").to(device)

    loss_func = nn.CrossEntropyLoss()

    results = []
    best_model = None
    for run in range(1):
        best_val_acc, best_test_acc, best_epoch, highest_test_acc = 100, 100, 0, 0

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr
        )
        for e in range(args.epochs):
            # --- train --- #
            
            loss = train(model, feats_train, y_train, loss_func, optimizer)
            tot_loss = loss
            correct, _ = evaluate(model, feats_valid, y_valid)
            val_acc = correct
            # print(val_acc)
            # --- test --- #
            test_correct, test_tot = 0, 0
            
            correct, _ = evaluate(model, feats_test, y_test)

            test_acc = correct
            if val_acc < best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = e + 1
            if test_acc < highest_test_acc:
                highest_test_acc = test_acc
            
            print(
                f"Epoch: {e+1:02d} "
                f"Loss: {tot_loss:.4f} "
                f"Valid mae: {val_acc:.4f} "
                f"Test mae: {test_acc:.4f}"
            )

        print(f"Run {run+1:02d}")
        print(f"Best epoch: {best_epoch}")
        print(f"Highest test mae: {highest_test_acc:.4f}")
        print(f"Valid mae: {best_val_acc:.4f}")
        print(f"Test mae: {best_test_acc:.4f}")

        results.append([highest_test_acc, best_val_acc, best_test_acc])

    results = torch.as_tensor(results)  # (runs, 3)
    print_str = f"{results.shape[0]} runs: "
    r = results[:, 0]
    print_str += f"Highest Test: {r.mean():.2f} ± {r.std():.2f} "
    r = results[:, 1]
    print_str += f"Best Valid: {r.mean():.2f} ± {r.std():.2f} "
    r = results[:, 2]
    print_str += f"Final Test: {r.mean():.2f} ± {r.std():.2f} "
    print_str += f"Best epoch: {best_epoch}"
    print(print_str)


if __name__ == "__main__":
    main()
