import os.path as osp
import time
import argparse
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear
from tqdm import tqdm
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import index_to_mask


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


def load_out_t(name):
    return torch.from_numpy(np.load(name)["arr_0"])

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--num_id', type=int, default=9)
parser.add_argument('--norm_type', type=str, default='none')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--id', type=str, default='ogbn-products')

args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
transform = T.Compose([T.ToSparseTensor()])
root = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
dataset = PygNodePropPredDataset('ogbn-products', root,
                                 transform=T.AddSelfLoops())
evaluator = Evaluator(name='ogbn-products')

data = dataset[0]

split_idx = dataset.get_idx_split()
for split in ['train', 'valid', 'test']:
    data[f'{split}_mask'] = index_to_mask(split_idx[split], data.y.shape[0])

semantic_id = load_out_t(f"semantic_ID_{args.id}.npz")
for i in range(semantic_id.shape[-1]):
    values = semantic_id[..., i].flatten()
    unique_values, counts = torch.unique(values, return_counts=True)
    print(f"location i: {list(zip(unique_values.tolist(), counts.tolist()))}")
print(semantic_id.shape)
step = 1
columns_to_select = [i for i in range(0, semantic_id.size(1), step)]
feats = torch.tensor(semantic_id).float()[:,columns_to_select]
# feats = dataset.graph['node_feat']
model_gnn = SimpleGNNLayer(args.k)
data_ = transform(data)
feats = model_gnn(feats, data_.adj_t).to(device)
data = data.to(device)
model = MLP(num_layers=args.num_layers,
            input_dim=args.num_id,
            hidden_dim=args.hidden_channels,
            output_dim=112,
            dropout_ratio=args.dropout,
            norm_type=args.norm_type).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()

    total_loss = total_examples = 0
    optimizer.zero_grad()
    
    feats_train = feats[data.train_mask]
    y = data.y[data.train_mask]
    batch_size = 500000

    num_batches = (feats_train.size(0) + batch_size - 1) // batch_size

    # all_outputs = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, feats.size(0))

        batch_feats = feats_train[start_idx:end_idx]

        batch_output = model(batch_feats)

        # all_outputs.append(batch_output)

    # out = torch.cat(all_outputs, dim=0)
    # out = model(feats)
    
        loss = F.cross_entropy(batch_output, y[start_idx:end_idx].view(-1))
        (loss).backward()
        optimizer.step()

    total_loss += float(loss) * int(data.train_mask.sum())
    total_examples += int(data.train_mask.sum())


    return total_loss / total_examples


@torch.no_grad()
def test(epoch):
    model.eval()

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}

    id_list_concat_all = []
    
    batch_size = 500000

    num_batches = (feats.size(0) + batch_size - 1) // batch_size

    all_outputs = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, feats.size(0))

        batch_feats = feats[start_idx:end_idx]

        batch_output = model(batch_feats)

        all_outputs.append(batch_output)

    out = torch.cat(all_outputs, dim=0).argmax(dim=-1, keepdim=True)

    
    # out = model(feats).argmax(dim=-1, keepdim=True)
    # print(out.shape,data.y.shape)
    for split in ['train', 'valid', 'test']:
        mask = data[f'{split}_mask']
        y_true[split].append(data.y[mask].cpu())
        y_pred[split].append(out[mask].cpu())

    train_acc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']

    test_acc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']

    return train_acc, valid_acc, test_acc


times = []
best_val = 0.0
final_train = 0.0
final_test = 0.0
for epoch in range(1, 1501):
    start = time.time()
    loss = train(epoch)
    train_acc, val_acc, test_acc = test(epoch)
    if val_acc > best_val:
        best_val = val_acc
        final_train = train_acc
        final_test = test_acc
    print(epoch, f'Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
    times.append(time.time() - start)

print(f'Final Train: {final_train:.4f}, Best Val: {best_val:.4f}, '
      f'Final Test: {final_test:.4f}')
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")