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
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import index_to_mask


class GNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        self.conv = SAGEConv(in_channels, out_channels)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = self.norm(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv(x, edge_index)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, kmeans=0, num_codes=16):
        super().__init__()

        self.dropout = dropout
        self.vqs = torch.nn.ModuleList()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)
        self.kmeans = kmeans
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GNNBlock(
                hidden_channels,
                hidden_channels,
            )
            self.convs.append(conv)
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

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin1(x)
        id_list = []
        quantized_list = []
        total_commit_loss = 0
        for (conv, vq) in zip(self.convs, self.vqs):
            x = conv(x, edge_index)
            if self.kmeans:
                quantized, _, commit_loss, dist, codebook = vq(x)
                id_list.append(torch.stack(_, dim=1))
                quantized_list.append(quantized)
                total_commit_loss += commit_loss
            else:
                x_, vq_ = vq(x)
                total_commit_loss += vq_['loss'].mean()
                id_list.append(vq_['q'])
        x = self.norm(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        id_list_concat = torch.cat(id_list, dim=1)
        return self.lin2(x), total_commit_loss, id_list_concat

parser = argparse.ArgumentParser()
parser.add_argument('--kmeans', type=int, default=0)
parser.add_argument('--num_codes', type=int, default=16)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
root = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
dataset = PygNodePropPredDataset('ogbn-products', root,
                                 transform=T.AddSelfLoops())
evaluator = Evaluator(name='ogbn-products')

data = dataset[0]
split_idx = dataset.get_idx_split()
for split in ['train', 'valid', 'test']:
    data[f'{split}_mask'] = index_to_mask(split_idx[split], data.y.shape[0])

train_loader = RandomNodeLoader(data, num_parts=10, shuffle=True,
                                num_workers=5)
# Increase the num_parts of the test loader if you cannot fit
# the full batch graph into your GPU:
test_loader = RandomNodeLoader(data, num_parts=1, num_workers=5)

model = GNN(
    in_channels=dataset.num_features,
    hidden_channels=128,
    out_channels=dataset.num_classes,
    num_layers=5,  # You can try 1000 layers for fun
    dropout=0.5,
    kmeans=args.kmeans, num_codes=args.num_codes
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:03d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()

        # Memory-efficient aggregations:
        data = transform(data)
        out, total_commit_loss, id_list_concat = model(data.x, data.adj_t)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].view(-1))
        (loss+total_commit_loss).backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())
        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(epoch):
    model.eval()

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:03d}')
    id_list_concat_all = []
    for data in test_loader:
        # Memory-efficient aggregations
        data = transform(data)
        out, total_commit_loss, id_list_concat = model(data.x, data.adj_t)
        out = out.argmax(dim=-1, keepdim=True)
        for split in ['train', 'valid', 'test']:
            mask = data[f'{split}_mask']
            id_list_concat_all.append(id_list_concat[mask])
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()
    id_list_concat_all = torch.cat(id_list_concat_all, dim=0)
    id_list_concat = id_list_concat.detach().cpu().numpy()
    np.savez(f"semantic_ID_ogbn-products", id_list_concat)
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
for epoch in range(1, 1001):
    start = time.time()
    loss = train(epoch)
    train_acc, val_acc, test_acc = test(epoch)
    if val_acc > best_val:
        best_val = val_acc
        final_train = train_acc
        final_test = test_acc
    print(f'Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
    times.append(time.time() - start)

print(f'Final Train: {final_train:.4f}, Best Val: {best_val:.4f}, '
      f'Final Test: {final_test:.4f}')
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")