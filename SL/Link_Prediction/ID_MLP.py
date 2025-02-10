import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
import time
from ogbdataset import loaddataset
from typing import Iterable


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class SimpleGNNLayer(MessagePassing):
    def __init__(self):
        super(SimpleGNNLayer, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        norm = deg.pow(-0.5).unsqueeze(1)
        for _ in range(10):
            x = x * norm
            x = self.propagate(edge_index, x=x)
            x = x * norm

        return x

    def message(self, x_j):
        return x_j
    
def train(
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None, feats = None):
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        # print(perm,perm.shape)
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
            
        h = feats
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss

        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

import pickle

@torch.no_grad()
def test(predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input, feats = None, dataset = None):

    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = feats
    

    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)


    pos_valid_pred = torch.cat([
        predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred = torch.cat([
        predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    if use_valedges_as_input:
        adj = data.full_adj_t
    #     h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)
    
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']

        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    return results, h.cpu()


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--dataset', type=str, default="collab")
    
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_id', type=int, default=15)
    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.3, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.3, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0003, help="learning rate of predictor")
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)

    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")

    
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', choices=convdict.keys())

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")

    
    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)
    args = parser.parse_args()
    return args

def load_out_t(name):
    return torch.from_numpy(np.load(name)["arr_0"])

def main():
    args = parseargs()
    print(args, flush=True)

    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
        evaluator = Evaluator(name=f'ogbl-ppa')
    else:
        evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    # device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load)
    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    if args.predictor in ["cn1"]:
        predfn = partial(predfn, use_xlin=False, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)

    ret = []

    for run in range(0, args.runs):
        set_seed(run+1)
        if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load) # get a new split of dataset
            data = data.to(device)
            # print(data,data.y)
            # s()
        bestscore = None
        
        
        predictor = predfn(args.num_id, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        # if args.loadmod:
            
        #     keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
        #     print("unmatched params", keys, flush=True)
        

        optimizer = torch.optim.Adam([ 
           {'params': predictor.parameters(), 'lr': args.prelr}])
        
        semantic_id = load_out_t(f"semantic_ID_{args.dataset}.npz")
        for i in range(semantic_id.shape[-1]):
            values = semantic_id[..., i].flatten()
            unique_values, counts = torch.unique(values, return_counts=True)
            print(f"location i: {list(zip(unique_values.tolist(), counts.tolist()))}")
        print(semantic_id)
        feats = torch.tensor(semantic_id).float().to(device)
        # feats = model_gnn(feats, data.edge_index)
        # feats = torch.rand(feats.shape).to(device)
        
        for epoch in range(1, 1 + args.epochs):
            alpha = None
            t1 = time.time()
            loss = train(predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha, feats)
            print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if True:
                t1 = time.time()
                results, h = test(predictor, data, split_edge, evaluator,
                               args.testbs, args.use_valedges_as_input, feats, dataset = args.dataset)
                print(f"test time {time.time()-t1:.2f} s")
                if bestscore is None:
                    bestscore = {key: list(results[key]) for key in results}
                for key, result in results.items():
                    writer.add_scalars(f"{key}_{run}", {
                        "trn": result[0],
                        "val": result[1],
                        "tst": result[2]
                    }, epoch)

                if True:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        if valid_hits > bestscore[key][1]:
                            bestscore[key] = list(result)
                            # if args.save_gemb:
                            #     torch.save(h, f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                            # if args.savemod:
                            #     torch.save(predictor.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---', flush=True)
        print(f"best {bestscore}")
        if args.dataset == "collab":
            ret.append(bestscore["Hits@50"][-2:])
        elif args.dataset == "ppa":
            ret.append(bestscore["Hits@100"][-2:])
        elif args.dataset == "ddi":
            ret.append(bestscore["Hits@20"][-2:])
        elif args.dataset == "citation2":
            ret.append(bestscore[-2:])
        elif args.dataset in ["Pubmed", "Cora", "Citeseer"]:
            ret.append(bestscore["Hits@100"][-2:])
        else:
            raise NotImplementedError
    ret = np.array(ret)
    print(ret)
    print(f"Final result: val {np.average(ret[:, 0]):.4f} {np.std(ret[:, 0]):.4f} tst {np.average(ret[:, 1]):.4f} {np.std(ret[:, 1]):.4f}")


if __name__ == "__main__":
    main()