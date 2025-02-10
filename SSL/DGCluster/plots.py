import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import copy
import torch

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

dataset_names = ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'coauthorcs', 'coauthorphysics']

dataset_name_map = {
    'cora': "Cora",
    'citeseer': "CiteSeer",
    'pubmed': "PubMed",
    'computers': "Amazon PC",
    'photo': "Amazon Photo",
    'coauthorcs': "Coauthor CS",
    'coauthorphysics': "Coauthor PHY"
}

markers = {
    "cora": "v",
    "citeseer": "^",
    "pubmed": "<",
    "computers": ">",
    "photo": "P",
    "coauthorcs": "X",
    "coauthorphysics": "D",
}

colors = {
    "cora": "r",
    "citeseer": "m",
    "pubmed": "g",
    "computers": "c",
    "photo": "b",
    "coauthorcs": "y",
    "coauthorphysics": "k",
}

evaluation_keys = ['modularity', 'nmi']

results = {dataset: {} for dataset in dataset_names}
for dataset in dataset_names:
    for evaluation_key in evaluation_keys:
        result_dataset_eval = []
        result_dataset_eval_std = []
        for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            results_dataset = []
            for seed in range(10):
                path = f"results/results_{dataset}_{lam}_300_gcn_{seed}.pt"
                results_dataset.append(torch.load(path)[evaluation_key])
            result_dataset_eval.append(np.mean(results_dataset))
            result_dataset_eval_std.append(np.std(results_dataset))
        results[dataset][evaluation_key] = copy.deepcopy(result_dataset_eval)
        results[dataset][evaluation_key + '_std'] = copy.deepcopy(result_dataset_eval_std)

fig, axes = plt.subplots(nrows=1, ncols=len(evaluation_keys), figsize=(len(evaluation_keys) * 4, 4), sharex=True)

labelsize = 14
ticksize = 12
markersize = 6
linewidth = 1.5

xticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

title_map = {
    'conductance': r'$\mathcal{C}$',
    'modularity': r'$\mathcal{Q}$',
    'nmi': r'NMI',
    'sample_f1_score': r'F1'
}

for i, key in enumerate(evaluation_keys):
    ax = axes[i]
    ls = [None] * len(dataset_names)
    for i, method_key in enumerate(dataset_names):
        ls[i], = ax.plot(xticks, results[method_key][key], label=method_key, marker=markers[method_key], color=colors[method_key])
        ax.fill_between(xticks, np.array(results[method_key][key]) - np.array(results[method_key][key + '_std']), np.array(results[method_key][key]) + np.array(results[method_key][key + '_std']), alpha=0.2, color=colors[method_key])

    ax.minorticks_off()
    ax.set_xticks(xticks)  # , [1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(xticks)
    ax.set_title(title_map[key], fontsize=labelsize)
    ax.tick_params(axis='x', labelsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)
    ax.grid(True)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'$\lambda$', fontsize=labelsize)

fig.tight_layout()
fig.subplots_adjust(left=0.035, bottom=0.16, right=0.99, wspace=0.22)

axes[0].legend(handles=ls, labels=[dataset_name_map[dataset_name] for dataset_name in dataset_names],
               loc='upper center', bbox_to_anchor=(1.1, -0.2), fancybox=False, shadow=False, ncol=math.ceil(len(dataset_names) / 2), fontsize=labelsize)

fig.savefig(f'plots/results_{"_".join(evaluation_keys)}.pdf', bbox_inches='tight')
plt.show()
