import argparse

import torch
import torch.nn.functional as F
import os
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger_ import Logger
import numpy as np
import nxmetis
import networkx as nx
import numpy as np
from torch_geometric.utils import to_undirected
import pickle
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class SimpleGNNLayer(MessagePassing):
    def __init__(self, k=10):
        super(SimpleGNNLayer, self).__init__(aggr='add')
        self.k = k
    def forward(self, x, adj):
       
        adj = adj.set_diag()
        deg = adj.sum(dim=1).squeeze()  

        norm = deg.pow(-0.5)
        norm = torch.where(torch.isinf(norm), torch.zeros_like(norm), norm)  
        norm = norm.unsqueeze(-1)

        for _ in range(self.k):
            x = x * norm
            x = matmul(adj, x, reduce="add")  
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

def train(model, y, feats, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out = model(feats)[train_idx]
    loss = criterion(out, y[train_idx].to(torch.float))
    (loss).backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, y, feats, split_idx, evaluator):
    model.eval()

    out = model(feats)
    # print(y.shape,out.shape)
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': out[split_idx['train']],
    })['rocauc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': out[split_idx['valid']],
    })['rocauc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': out[split_idx['test']],
    })['rocauc']

    return train_acc, valid_acc, test_acc


def load_out_t(name):
    return torch.from_numpy(np.load(name)["arr_0"])
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--num_id', type=int, default=3)
    parser.add_argument('--norm_type', type=str, default='none')
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]
    
    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)
    
    print(data.num_features,data.x.shape)
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    semantic_id = load_out_t(f"semantic_ID_ogbn-proteins.npz")
    for i in range(semantic_id.shape[-1]):
        values = semantic_id[..., i].flatten()
        unique_values, counts = torch.unique(values, return_counts=True)
        print(f"location i: {list(zip(unique_values.tolist(), counts.tolist()))}")
    print(semantic_id.shape)
    feats = torch.tensor(semantic_id).float()
    # feats = dataset.graph['node_feat']
    model_gnn = SimpleGNNLayer(args.k)
    feats = model_gnn(feats, data.adj_t).to(device)

    data = data.to(device)
    model = MLP(num_layers=args.num_layers,
            input_dim=args.num_id,
            hidden_dim=args.hidden_channels,
            output_dim=112,
            dropout_ratio=args.dropout,
            norm_type=args.norm_type).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data.y, feats, train_idx, optimizer)
            if epoch % args.eval_steps == 0:
                result = test(model, data.y, feats, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
