import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.metrics import f1_score, roc_auc_score
from torch_sparse import SparseTensor


def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """randomly splits label into train/valid/test splits"""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    """use all remaining data points as test data, so test_num will not be used"""
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = (
        non_train_idx[:valid_num],
        non_train_idx[valid_num : valid_num + test_num],
    )
    print(f"train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")
    split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    return split_idx


def normalize_feat(mx):
    """Row-normalize np or sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t

def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        if args.method == "fast_transgnn" or args.method == "glcn":
            out, _ = model(dataset)
        else:
            out, total_commit_loss, id_list_concat, x_ = model(dataset)
    # print(id_list_concat.shape)
    # x_= x_.detach().cpu().numpy()
    # np.savez(f"GNN_ID_{args.dataset}", x_)
    
    id_list_concat = id_list_concat.detach().cpu().numpy()
    np.savez(f"semantic_ID_{args.dataset}", id_list_concat)
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


def load_fixed_splits(dataset, name, protocol):

    splits_lst = []
    if name in ["cora", "citeseer", "pubmed"] and protocol == "semi":
        splits = {}
        splits["train"] = torch.as_tensor(dataset.train_idx)
        splits["valid"] = torch.as_tensor(dataset.valid_idx)
        splits["test"] = torch.as_tensor(dataset.test_idx)
        splits_lst.append(splits)
    elif name in ["chameleon", "squirrel"]:
        file_path = f"./data/geom-gcn/{name}/{name}_filtered.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # (10, N), 10 splits
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        for i in range(10):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"] = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)

    elif name in ["film"]:
        for i in range(10):
            splits_file_path = (
                "./data/geom-gcn/splits/{}".format(name, name)
                + "_split_0.6_0.2_"
                + str(i)
                + ".npz"
            )
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits["train"] = torch.BoolTensor(splits_file["train_mask"])
                splits["valid"] = torch.BoolTensor(splits_file["val_mask"])
                splits["test"] = torch.BoolTensor(splits_file["test_mask"])
            splits_lst.append(splits)
    elif name in ['pokec']:
        dir = f'./data/pokec/split_0.5_0.25'
        tensor_split_idx = {}
        if os.path.exists(dir):
            tensor_split_idx['train'] = torch.as_tensor(np.loadtxt(dir + '/pokec_train.txt'), dtype=torch.long)
            tensor_split_idx['valid'] = torch.as_tensor(np.loadtxt(dir + '/pokec_valid.txt'), dtype=torch.long)
            tensor_split_idx['test'] = torch.as_tensor(np.loadtxt(dir + '/pokec_test.txt'), dtype=torch.long)
        splits_lst.append(tensor_split_idx)
    else:
        raise NotImplementedError

    return splits_lst

dataset_drive_url = {
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
}
