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


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_codes=16, kmeans=0, jk=False):
        super(SAGE, self).__init__()

        self.vqs = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.kmeans = kmeans
        if self.kmeans:
            from vq import VectorQuantize, ResidualVectorQuant
            print("kmeans")
            self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=num_codes, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
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

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.kmeans:
                self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=num_codes, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
            else:
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

        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        if self.kmeans:
            self.vqs.append(ResidualVectorQuant(dim=hidden_channels, codebook_size=num_codes, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False))
        else:
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

        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.jk = jk
        if self.jk:
            self.register_parameter("jkparams", torch.nn.Parameter(torch.randn((num_layers,))))
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        id_list = []
        quantized_list = []
        total_commit_loss = 0
        jkx = []
        for i, (conv, vq) in enumerate(zip(self.convs[:-1], self.vqs[:-1])):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            jkx.append(x)
            if self.kmeans:
                quantized, _, commit_loss, dist, codebook = vq(x)
                id_list.append(torch.stack(_, dim=1))
                quantized_list.append(quantized)
                total_commit_loss += commit_loss
            else:
                x_, vq_ = vq(x)
                total_commit_loss += vq_['loss'].mean()
                id_list.append(vq_['q'])
        x = self.convs[-1](x, adj_t)
        jkx.append(x)
        if self.kmeans:
            quantized, _, commit_loss, dist, codebook = self.vqs[-1](x)
            id_list.append(torch.stack(_, dim=1))
            quantized_list.append(quantized)
            total_commit_loss += commit_loss
        else:
            x_, vq_ = self.vqs[-1](x)
            total_commit_loss += vq_['loss'].mean()
            id_list.append(vq_['q'])
        if self.jk: # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        id_list_concat = torch.cat(id_list, dim=1)
        return self.linear(x), total_commit_loss, id_list_concat


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out, total_commit_loss, id_list_concat = model(data.x, data.adj_t)
    loss = criterion(out[train_idx], data.y[train_idx].to(torch.float))
    (loss+total_commit_loss).backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, args):
    model.eval()

    out, total_commit_loss, id_list_concat = model(data.x, data.adj_t)
    id_list_concat = id_list_concat.detach().cpu().numpy()
    np.savez(f"semantic_ID_ogbn-proteins", id_list_concat)
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': out[split_idx['train']],
    })['rocauc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': out[split_idx['valid']],
    })['rocauc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': out[split_idx['test']],
    })['rocauc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--gnum_layers', type=int, default=3)
    parser.add_argument('--ghidden_channels', type=int, default=256)
    parser.add_argument('--gdropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--kmeans', type=int, default=1)
    parser.add_argument('--num_codes', type=int, default=16)

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


    data = data.to(device)
    model = SAGE(in_channels=data.num_features, hidden_channels=args.ghidden_channels,
                        out_channels=112, num_layers=args.gnum_layers,
                        dropout=args.gdropout, num_codes=args.num_codes, kmeans=args.kmeans, jk=args.jk).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator, args)
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