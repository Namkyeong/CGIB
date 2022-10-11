import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import BRICS

def write_experiment(args, config_str, best_config):
    
    f = open("results/{}/{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write(best_config)
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()

def write_summary_cv(args, config_str, rmse, rmse_std, best_mses, mae, mae_std, best_maes):

    f = open("results/{}/cv_{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("5 fold results --> RMSE : {:.4f}({:.4f}) || MAE : {:.4f}({:.4f}) ".format(rmse, rmse_std, mae, mae_std))
    f.write("\n")
    f.write("Individual folds result --> RMSE : {:.4f} || {:.4f} || {:.4f} || {:.4f} || {:.4f}".format(best_mses[0], best_mses[1], best_mses[2], best_mses[3], best_mses[4]))
    f.write("\n")
    f.write("Individual folds result --> MAE : {:.4f} || {:.4f} || {:.4f} || {:.4f} || {:.4f}".format(best_maes[0], best_maes[1], best_maes[2], best_maes[3], best_maes[4]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()

def write_summary(args, config_str, rmse, rmse_std, mae, mae_std):

    f = open("results/{}/summary_{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("{} run results --> RMSE : {:.4f}({:.4f}) || MAE : {:.4f}({:.4f})".format(args.repeat, rmse, rmse_std, mae, mae_std))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_len_matrix(len_list):
    len_list = np.array(len_list)
    max_nodes = np.sum(len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum + l] = 1
        len_matrix.append(curr)
        curr_sum += l
    return np.array(len_matrix)


def create_batch_mask(samples):
    batch0 = samples[0].batch.reshape(1, -1)
    index0 = torch.cat([batch0, torch.tensor(range(batch0.shape[1])).reshape(1, -1)])
    mask0 = torch.sparse_coo_tensor(index0, torch.ones(index0.shape[1]), size = (batch0.max() + 1, batch0.shape[1]))

    batch1 = samples[1].batch.reshape(1, -1)
    index1 = torch.cat([batch1, torch.tensor(range(batch1.shape[1])).reshape(1, -1)])
    mask1 = torch.sparse_coo_tensor(index1, torch.ones(index1.shape[1]), size = (batch1.max() + 1, batch1.shape[1]))

    return mask0, mask1