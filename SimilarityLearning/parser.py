import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run the code.")

    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--embedder', type=str, default='CGIB', help='Specify the model')
    parser.add_argument("--dataset", nargs="?", default="AIDS700nef", help="Dataset name. reg: AIDS700nef/LINUX/IMDBMulti and cls: ffmpeg_min3/20/50, openssl_min3/20/50")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs. Default is 10000.")
    parser.add_argument("--batch-size", type=int, default=512, help="Number of graph pairs per batch. Default is 128.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability. Default is 0.0.")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Adam weight decay. Default is 5*10^-4.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Specify cuda devices')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')

    parser.add_argument("--nhid", type=int, default=100,help="Hidden dimension in convolution. Default is 64.")
    
    if parser.parse_known_args()[0].embedder == "H2MN":
        parser.add_argument("--ratio1", type=float, default=0.8, help="Pooling rate. Default is 0.8.")
        parser.add_argument("--ratio2", type=float, default=0.8, help="Pooling rate. Default is 0.8.")
        parser.add_argument("--ratio3", type=float, default=0.8, help="Pooling rate. Default is 0.8.")
        
        parser.add_argument('--mode', type=str, default='RW', help='Specify hypergraph construction mode NEighbor(NE)/RandomWalk(RW)')  
        parser.add_argument('--k', type=int, default=5, help='Hyperparameter for construction hyperedge')
    
    if parser.parse_known_args()[0].embedder == "CGIB":
        parser.add_argument("--beta", type=float, default=1.0)

    if parser.parse_known_args()[0].embedder == "CGIB_cont":
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--tau", type=float, default=1.0)

    return parser.parse_args()


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['seed', 'epochs', 'nhid', 'batch_size', 'dropout', 'ratio1', 'ratio2', 'ratio3', 'weight_decay', 'device', 'patience', 'num_features']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]