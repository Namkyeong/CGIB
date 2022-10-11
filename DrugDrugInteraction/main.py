import pandas as pd
import torch
import random
import numpy as np
import os
import argument
import time
from utils import get_stats, write_summary, write_summary_total

torch.set_num_threads(2)

def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def experiment():

    args, unknown = argument.parse_args()
    
    print("Loading dataset...")
    start = time.time()

    # Load dataset
    train_set = torch.load("./data/processed/{}_train.pt".format(args.dataset))
    valid_set = torch.load("./data/processed/{}_valid.pt".format(args.dataset)) 
    test_set = torch.load("./data/processed/{}_test.pt".format(args.dataset))

    print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))

    best_rocs, best_aps, best_f1s, best_accs = [], [], [], []
    
    for repeat in range(1, args.repeat + 1):
    
        stats, config_str, _, _ = main(args, train_set, valid_set, test_set, repeat = repeat)
        
        # get Stats
        best_rocs.append(stats[0])
        best_aps.append(stats[1])
        best_f1s.append(stats[2])
        best_accs.append(stats[3])

        write_summary(args, config_str, stats)
    
    roc_mean, roc_std = get_stats(best_rocs)
    ap_mean, ap_std = get_stats(best_aps)
    f1_mean, f1_std = get_stats(best_f1s)
    accs_mean, accs_std = get_stats(best_accs)

    write_summary_total(args, config_str, [roc_mean, roc_std, ap_mean, ap_std, f1_mean, f1_std, accs_mean, accs_std])
    
    

def main(args, train_df, valid_df, test_df, repeat = 0, fold = 0):

    if args.embedder == 'CGIB':
        from models import CGIB_ModelTrainer
        embedder = CGIB_ModelTrainer(args, train_df, valid_df, test_df, repeat, fold)

    elif args.embedder == 'CGIB_cont':
        from models import CGIB_cont_ModelTrainer
        embedder = CGIB_cont_ModelTrainer(args, train_df, valid_df, test_df, repeat, fold)

    best_roc, best_ap, best_f1, best_acc = embedder.train()

    return [best_roc, best_ap, best_f1, best_acc], embedder.config_str, embedder.best_config_roc, embedder.best_config_f1


if __name__ == "__main__":
    experiment()


