import os
import time
import glob

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from scipy.stats import spearmanr, kendalltau

from models import CGIB, CGIB_cont
from parser import parameter_parser, config2string
from utils import tab_printer, GraphRegressionDataset, prec_at_ks, calculate_ranking_correlation, create_batch_mask

from tensorboardX import SummaryWriter

args = parameter_parser()
dataset = GraphRegressionDataset(args)
args.num_features = dataset.number_features

config_str = config2string(args)
writer = SummaryWriter(log_dir="runs/{}".format(config_str))

os.environ["CUDA_VISIBLE_DEVICES"] = args.device[-1]
device = f'{args.device}' if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
tab_printer(args)

if args.embedder == "CGIB":
    model = CGIB(args, device).to(args.device)
elif args.embedder == "CGIB_cont":
    model = CGIB_cont(args, device).to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def train():
    print('\nModel training.\n')
    start = time.time()
    val_loss_values = []
    patience_cnt = 0
    best_epoch = 0
    min_loss = 1e10

    with torch.autograd.detect_anomaly():
        for epoch in range(args.epochs):
            model.train()
            batches = dataset.create_batches(dataset.training_set)
            main_index = 0
            loss_sum = 0

            for index, batch_pair in enumerate(batches):
                optimizer.zero_grad()
                data = dataset.transform(batch_pair)
                masks = create_batch_mask(data, device)

                prediction = model(data, masks)
                loss = F.mse_loss(prediction, data['target'], reduction='sum')

                prediction, KL_Loss, pair_pred_loss = model(data, masks, bottleneck = True)
                loss += F.mse_loss(prediction, data['target'], reduction='sum')
                loss += args.beta * (KL_Loss + pair_pred_loss)
                
                loss.backward()
                optimizer.step()
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss.item()
            loss = loss_sum / main_index

            # start validate at 9000th iteration
            val_loss = validate()
            end = time.time()
            print('Epoch: {:05d},'.format(epoch + 1), 'loss_train: {:.6f},'.format(loss), 'loss_val: {:.6f},'.format(val_loss), 'time: {:.6f}s'.format(end - start))
            writer.add_scalar("accs/train_loss", loss, epoch)
            writer.add_scalar("accs/valid_RMSE", val_loss, epoch)
            val_loss_values.append(val_loss)
            
            if epoch + 1 > 9000:
                if val_loss_values[-1] < min_loss:
                    min_loss = val_loss_values[-1]
                    best_epoch = epoch
                    torch.save(model.state_dict(), './model_checkpoints/{}.pth'.format(config_str))
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt == args.patience:
                    break

        print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - start))

        return best_epoch


def validate():
    model.eval()
    batches = dataset.create_batches(dataset.val_set)
    main_index = 0
    loss_sum = 0
    with torch.no_grad():
        for index, batch_pair in enumerate(batches):
            data = dataset.transform(batch_pair)
            masks = create_batch_mask(data, device)
            prediction = model(data, masks, test = True)
            loss = F.mse_loss(prediction, data['target'], reduction='sum')
            main_index = main_index + batch_pair[0].num_graphs
            loss_sum = loss_sum + loss.item()
        loss = loss_sum / main_index

    return loss


def evaluate():
    print('\nModel evaluation.')
    model.eval()
    scores = np.zeros((len(dataset.testing_graphs), len(dataset.training_graphs)))
    ground_truth = np.zeros((len(dataset.testing_graphs), len(dataset.training_graphs)))
    prediction_mat = np.zeros((len(dataset.testing_graphs), len(dataset.training_graphs)))

    rho_list = []
    tau_list = []
    prec_at_10_list = []
    prec_at_20_list = []

    with torch.no_grad():
        for i, g in enumerate(dataset.testing_graphs):
            if len(dataset.training_graphs) <= args.batch_size:
                source_batch = Batch.from_data_list([g] * len(dataset.training_graphs))
                target_batch = Batch.from_data_list(dataset.training_graphs)

                data = dataset.transform((source_batch, target_batch))
                target = data['target']
                ground_truth[i] = target.cpu().numpy()
                prediction = model(data)
                prediction_mat[i] = prediction.detach().cpu().numpy()

                scores[i] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()

                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
                prec_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))
                prec_at_20_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 20))
            else:
                # Avoid GPU OOM error
                batch_index = 0
                target_loader = DataLoader(dataset.training_graphs, batch_size=args.batch_size, shuffle=False)
                for index, target_batch in enumerate(target_loader):
                    source_batch = Batch.from_data_list([g] * target_batch.num_graphs)
                    data = dataset.transform((source_batch, target_batch))
                    masks = create_batch_mask(data, device)
                    target = data['target']
                    num_graphs = target_batch.num_graphs
                    ground_truth[i,batch_index: batch_index+num_graphs] = target.cpu().numpy()
                    prediction = model(data, masks, test = True)
                    prediction_mat[i,batch_index: batch_index+num_graphs] = prediction.detach().cpu().numpy()
                    scores[i,batch_index: batch_index+num_graphs] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()
                    batch_index += num_graphs
                
                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                prec_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))

    rho = np.mean(rho_list)
    prec_at_10 = np.mean(prec_at_10_list)
    model_error = np.mean(scores) * 0.5
    
    model_error, rho, prec_at_10 = print_evaluation(model_error, rho, prec_at_10)

    return model_error, rho, prec_at_10


def print_evaluation(model_error, rho, prec_at_10):
    
    model_error = model_error * 1000

    print("\nmse(10^-3): " + str(round(model_error, 5)))
    print("Spearman's rho: " + str(round(rho, 5)))
    print("p@10: " + str(round(prec_at_10, 5)))

    return model_error, rho, prec_at_10


def get_stats(array):
    
    mean = np.mean(np.asarray(array))
    std = np.std(np.asarray(array))

    return mean, std


def write_summary(args, config_str, stats):
    
    f = open("results/{}/{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("MSE : {:.4f} || rho : {:.4f} || prec : {:.4f} ".format(stats[0], stats[1], stats[2]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def write_summary_total(args, config_str, stats):
    
    f = open("results/{}/{}_total.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("MSE : {:.4f}({:.4f}) || rho : {:.4f}({:.4f}) || prec : {:.4f}({:.4f}) ".format(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


if __name__ == "__main__":
    
    best_mses, best_rhos, best_precs = [], [], []

    for i in range(5):
        best_model = train()
        model.load_state_dict(torch.load('./model_checkpoints/{}.pth'.format(config_str)))
        model_error, rho, prec_at_10 = evaluate()
        
        best_mses.append(model_error)
        best_rhos.append(rho)
        best_precs.append(prec_at_10)

        write_summary(args, config_str, [model_error, rho, prec_at_10])

    mse_mean, mse_std = get_stats(best_mses)
    rho_mean, rho_std = get_stats(best_rhos)
    prec_mean, prec_std = get_stats(best_precs)

    write_summary_total(args, config_str, [mse_mean, mse_std, rho_mean, rho_std, prec_mean, prec_std])
