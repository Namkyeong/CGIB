
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Union
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, f1_score, accuracy_score

def get_stats(array):
    
    mean = np.mean(np.asarray(array))
    std = np.std(np.asarray(array))

    return mean, std


def write_summary(args, config_str, stats):
    
    f = open("results/{}/{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("ROC : {:.4f} || AP : {:.4f} || F1 : {:.4f} || Acc : {:.4f} ".format(stats[0], stats[1], stats[2], stats[3]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def write_summary_total(args, config_str, stats):
    
    f = open("results/{}/{}_total.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("ROC : {:.4f}({:.4f}) || AP : {:.4f}({:.4f}) || F1 : {:.4f}({:.4f}) || Acc : {:.4f}({:.4f}) ".format(stats[0], stats[1], stats[2], stats[3],
                                                                                                                                    stats[4], stats[5], stats[6], stats[7]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def write_experiment(args, config_str, best_config):
    
    f = open("results/{}/{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write(best_config)
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


class KLD(nn.Module):
    def forward(self, inputs, targets):

        inputs = F.log_softmax(inputs, dim=0)
        targets = F.softmax(targets, dim=0)
        
        return F.kl_div(inputs, targets, reduction='batchmean')


def get_roc_score(preds, labels):

    preds_all, preds_all_ = eval_threshold(labels, preds)

    roc_score = roc_auc_score(labels, preds_all)
    ap_score = average_precision_score(labels, preds_all)
    f1_score_ = f1_score(labels, preds_all_)
    acc_score = accuracy_score(labels, preds_all_)
    return roc_score, ap_score, f1_score_, acc_score


def eval_threshold(labels_all, preds_all):

    # fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    optimal_threshold = 0.5
    preds_all_ = []
    
    for p in preds_all:
        if p >=optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)

    return preds_all, preds_all_