from models import *


def parse_method(method, args, c, d, device):
    if method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn, num_codes=args.num_codes, kmeans=args.kmeans).to(device)
    else:
        raise ValueError(f'Invalid method {method}')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.6,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=500,
                        help='Total number of test')
    
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--batch_size', type=int, default=100000, help='mini batch training for large graphs')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')
    parser.add_argument('--kmeans', type=int, default=0)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--num_id', type=int, default=9)
    parser.add_argument('--norm_type', type=str, default='none')
    parser.add_argument('--num_codes', type=int, default=16)
    
    # model
    parser.add_argument('--method', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true',
                        help='use residual link for each GNN layer')
    parser.add_argument('--use_weight', action='store_true',
                        help='use weight for GNN convolution')
    parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--patience', type=int, default=200,
                        help='early stopping patience.')
    
    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')

    parser.add_argument('--no_feat_norm', action='store_true',
                        help='Not use feature normalization.')
