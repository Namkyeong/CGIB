import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--embedder", type=str, default="CGIB")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default = 0.0)
    parser.add_argument("--dropout", type=float, default = 0.0)
    parser.add_argument("--scheduler", type=str, default="plateau", help = "plateau")
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--es", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--message_passing", type=int, default=3)
    parser.add_argument("--cv", action = "store_true", default=False, help = 'cross validation when True')
    parser.add_argument("--save_checkpoints", action = "store_true", default=False, help = 'cross validation when True')
    parser.add_argument("--writer", action = "store_true", default=False, help = 'Tensorboard writer')

    # train dataset
    parser.add_argument("--dataset", type = str, default = 'ZhangDDI', help = 'ZhangDDI / ChChMiner / DeepDDI')
    
    # Hyperparameters for CGIB
    parser.add_argument("--beta", type=float, default=0.0001)
    
    if 'cont' in parser.parse_known_args()[0].embedder:
        parser.add_argument("--tau", type=float, default=1.0)

    return parser.parse_known_args()


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['fold', 'repeat', 'root', 'task', 'eval_freq', 'patience', 'device', 'writer',
                        'scheduler', 'fg_pooler', 'prob_temp', 'es', 'epochs', 'cv', 'interaction', 
                        'norm_loss', 'layers', 'pred_hid', 'mad', "anneal_rate", "temp_min", 
                        "sparsity_regularizer", "entropy_regularizer", "message_passing"]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]