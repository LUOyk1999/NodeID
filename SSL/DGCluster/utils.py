import random
import torch
import numpy as np

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score

import networkx as nx


def sample_f1_score(test_data, clusters, num_nodes):
    k = 10
    res = 0
    for i in range(k):
        s = random.sample(range(0, num_nodes), 1000)

        mx = max(clusters)
        s_clusters = clusters[s]

        MM = np.zeros((len(s_clusters), mx + 1))
        for i in range(len(s_clusters)):
            MM[i][s_clusters[i]] = 1
        MM = torch.tensor(MM)
        MM = torch.matmul(MM, torch.t(MM)).flatten()

        labels = test_data.y.squeeze()
        mx = max(labels)

        s_labels = labels[s]

        CM = np.zeros((len(s_labels), mx + 1))
        for i in range(len(s_labels)):
            CM[i][s_labels[i]] = 1
        CM = torch.tensor(CM)
        CM = torch.matmul(CM, torch.t(CM)).flatten()

        x = f1_score(CM, MM)
        res = res + x

    return res / k


def compute_fast_modularity(clusters, num_nodes, num_edges, torch_sparse_adj, degree, device):
    mx = max(clusters)
    MM = np.zeros((num_nodes, mx + 1))
    for i in range(len(clusters)):
        MM[i][clusters[i]] = 1
    MM = torch.tensor(MM).double().to(device)

    x = torch.matmul(torch.t(MM), torch_sparse_adj.double())
    x = torch.matmul(x, MM)
    x = torch.trace(x)

    y = torch.matmul(torch.t(MM), degree.double())
    y = torch.matmul(torch.t(y.unsqueeze(dim=0)), y.unsqueeze(dim=0))
    y = torch.trace(y)
    y = y / (2 * num_edges)
    return ((x - y) / (2 * num_edges)).item()


def compute_nmi(clusters, labels):
    return normalized_mutual_info_score(clusters, labels)


def compute_conductance(clusters, Graph):
    comms = [[] for i in range(max(clusters) + 1)]
    for i in range(len(clusters)):
        comms[clusters[i]].append(i)
    conductance=[]
    for com in comms:
        try:
            conductance.append(nx.conductance(Graph, com, weight='weight'))
        except:
            continue

    return conductance
