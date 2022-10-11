import torch
import random
import numpy as np
import os
import argument
from utils import write_summary, write_summary_cv

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
    
    dataset = torch.load("./data/processed/{}.pt".format(args.dataset))
    len_data = len(dataset)

    random.shuffle(dataset)
    len_test = int(len_data / args.fold)
    len_val = int(len_test * 0.5)

    dataset = np.asarray(dataset)
    
    cv_rmses = []
    cv_maes = []

    for repeat in range(args.repeat):
        
        # Set Random Seed
        seed_everything(repeat)
        best_mses = []
        best_maes = []

        for fold in range(args.fold):

            test_index = np.asarray([i for i in range(fold * len_test, (fold + 1) * len_test)])
            train_index = np.asarray([i for i in range(len_data) if np.isin(i, test_index, invert = True)])
            
            test_set = dataset[test_index].tolist()
            train_set = dataset[train_index].tolist()
            
            # Get validation set
            val_set = test_set[:len_val]
            test_set = test_set[len_val:]
            
            best_mse, best_mae, config_str, best_config = main(args, train_set, val_set, test_set, repeat, fold)

            best_mses.append(best_mse)
            best_maes.append(best_mae)
            
        rmse = np.mean(np.sqrt(np.asarray(best_mses)))
        rmse_std = np.std(np.sqrt(np.asarray(best_mses)))

        mae = np.mean(np.asarray(best_maes))
        mae_std = np.std(np.asarray(best_maes))

        cv_rmses.append(rmse)
        cv_maes.append(mae)
        write_summary_cv(args, config_str, rmse, rmse_std, np.sqrt(np.asarray(best_mses)), mae, mae_std, np.asarray(best_maes))

    rmse = np.mean(np.asarray(cv_rmses))
    rmse_std = np.std(np.asarray(cv_rmses))

    mae = np.mean(np.asarray(cv_maes))
    mae_std = np.std(np.asarray(cv_maes))

    write_summary(args, config_str, rmse, rmse_std, mae, mae_std)


def main(args, train_df, valid_df, test_df, repeat, fold = 0):

    if args.embedder == 'CGIB':
        from models import CGIB_ModelTrainer
        embedder = CGIB_ModelTrainer(args, train_df, valid_df, test_df, repeat, fold)

    elif args.embedder == 'CGIB_cont':
        from models import CGIB_cont_ModelTrainer
        embedder = CGIB_cont_ModelTrainer(args, train_df, valid_df, test_df, repeat, fold)

    best_mse, best_mae = embedder.train()

    return best_mse, best_mae, embedder.config_str, embedder.best_config


if __name__ == "__main__":
    experiment()


