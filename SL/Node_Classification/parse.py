from model import GNN

def parse_method(args, n, c, d, device):
    if args.method == 'gcn':
        model = GNN(d, args.hidden_channels, c, local_layers=args.local_layers,
                in_dropout=args.in_dropout, dropout=args.dropout,
                heads=args.num_heads, pre_ln=args.pre_ln, kmeans=args.kmeans, num_codes=args.num_codes, gnn='gcn').to(device)
    else:
        model = GNN(d, args.hidden_channels, c, local_layers=args.local_layers,
                in_dropout=args.in_dropout, dropout=args.dropout,
                heads=args.num_heads, pre_ln=args.pre_ln, kmeans=args.kmeans, num_codes=args.num_codes, gnn='gat').to(device)
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')

    parser.add_argument('--train_prop', type=float, default=.6,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    # model
    parser.add_argument('--method', type=str, default='gat')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7,
                        help='number of layers for local attention')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')
        
    parser.add_argument('--kmeans', type=int,
                        default=1)
    parser.add_argument('--num_codes', type=int,
                        default=16)
    parser.add_argument('--norm_type', type=str, default='none')

    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--num_id', type=int, default=15)
