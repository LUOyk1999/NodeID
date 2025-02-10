import torch
import numpy as np

dataset_names = ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'coauthorcs', 'coauthorphysics']

# Table 1 printing
print('Table 1 Results')
print('-----------------------------------')
for lam in [0.0, 0.2, 0.8]:
    performance = "\\model (\\lambda={}) & ".format(lam)
    for dataset_name in dataset_names:
        scores_1 = []
        scores_2 = []
        for seed in range(10):
            path = f'results/results_{dataset_name}_{lam}_300_gcn_{seed}.pt'
            res = torch.load(path)
            scores_1.append(res['conductance'])
            scores_2.append(res['modularity'])
        score_1 = np.mean(scores_1)
        score_2 = np.mean(scores_2)
        performance += f"{(score_1 * 100):.1f} & {(score_2 * 100):.1f} & "
    print(performance[:-2] + '\\\\')
print('-----------------------------------\n')

# Table 3 printing
print('Table 3 Results')
print('-----------------------------------')
for lam in [0.0, 0.2, 0.8]:
    performance = "\\model (\\lambda={}) & ".format(lam)
    for dataset_name in dataset_names:
        scores_1 = []
        scores_2 = []
        for seed in range(10):
            path = f'results/results_{dataset_name}_{lam}_300_gcn_{seed}.pt'
            res = torch.load(path)
            scores_1.append(res['nmi'])
            scores_2.append(res['sample_f1_score'])
        score_1 = np.mean(scores_1)
        score_2 = np.mean(scores_2)
        performance += f"{(score_1 * 100):.1f} & {(score_2 * 100):.1f} & "
    print(performance[:-2] + '\\\\')
print('-----------------------------------\n')

# Table 4 printing
print('Table 4 Results')
print('-----------------------------------')
dataset_name_map = {
    'cora': "Cora",
    'citeseer': "CiteSeer",
    'pubmed': "PubMed",
    'computers': "Amazon PC",
    'photo': "Amazon Photo",
    'coauthorcs': "Coauthor CS",
    'coauthorphysics': "Coauthor PHYSICS"
}
lam = 0.2
all_data = []
for dataset_name in dataset_names:
    row_data = []
    performance = f"\\textsc{{{dataset_name_map[dataset_name]}}} & "
    for metric in ['conductance', 'modularity', 'nmi', 'sample_f1_score']:
        for base_model in ['gcn', 'gat', 'gin', 'sage']:
            scores = []
            for seed in range(10):
                path = f'results/results_{dataset_name}_{lam}_300_{base_model}_{seed}.pt'
                res = torch.load(path)
                scores.append(res[metric])
            score = np.mean(scores)
            performance += f"{(score * 100):.1f} & "
            row_data.append(score)
    print(performance[:-2] + '\\\\')
    all_data.append(row_data)
all_data = np.array(all_data).mean(axis=0)
performance = f"\\textsc{{AVERAGE}} & "
for i in range(len(all_data)):
    performance += f"{(all_data[i] * 100):.1f} & "
print(performance[:-2] + '\\\\')
print('-----------------------------------\n')

# Table 5 printing
dataset_names_2 = ['cora', 'citeseer', 'pubmed']
print('Table 5 Results')
print('-----------------------------------')
for alp in [0.0, 0.5, 1.0]:
    performance = f"$\\alpha={alp}$ & "
    for dataset_name in dataset_names_2:
        scores_1 = []
        scores_2 = []
        for seed in range(10):
            if alp == 0.0:
                path = f'results/results_{dataset_name}_{lam}_300_gcn_{seed}.pt'
            else:
                path = f'results/results_{dataset_name}_{lam}_{alp}_300_gcn_{seed}.pt'
            res = torch.load(path)
            scores_1.append(res['modularity'])
            scores_2.append(res['sample_f1_score'])
        score_1 = np.mean(scores_1)
        score_2 = np.mean(scores_2)
        performance += f"{(score_1 * 100):.1f} & {(score_2 * 100):.1f} & "
    print(performance[:-2] + '\\\\')
print('-----------------------------------\n')
