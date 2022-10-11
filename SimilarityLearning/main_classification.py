import os
import time
import glob
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models import CGIB, CGIB_cont
from parser import parameter_parser, config2string
from utils import tab_printer, GraphClassificationDataset, create_batch_mask

from tensorboardX import SummaryWriter

args = parameter_parser()
dataset = GraphClassificationDataset(args)
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
            main_index = 0
            loss_sum = 0
            batches = dataset.create_batches(dataset.training_funcs, dataset.collate)

            for index, batch_pair in enumerate(batches):
                optimizer.zero_grad()
                data = dataset.transform(batch_pair)
                masks = create_batch_mask(data, device)

                prediction = model(data, masks)
                loss = F.binary_cross_entropy(prediction, data['target'], reduction='sum')

                prediction, KL_Loss, pair_pred_loss = model(data, masks, bottleneck = True)
                loss += F.binary_cross_entropy(prediction, data['target'], reduction='sum')
                loss += args.beta * (KL_Loss + pair_pred_loss)
                
                loss.backward()
                optimizer.step()
                main_index = main_index + len(batch_pair[2])
                loss_sum = loss_sum + loss.item()
            loss = loss_sum / main_index

            val_loss, aucscore = validate(dataset, dataset.validation_funcs)
            end = time.time()
            print('Epoch: {:05d},'.format(epoch + 1), 'loss_train: {:.6f},'.format(loss), 'loss_val: {:.6f},'.format(val_loss), 'AUC: {:.6f},'.format(aucscore), 'time: {:.6f}s'.format(end - start))
            writer.add_scalar("accs/train_loss", loss, epoch)
            writer.add_scalar("accs/valid_loss", val_loss, epoch)
            writer.add_scalar("accs/valid_AUC", aucscore, epoch)
            val_loss_values.append(val_loss)

            if epoch + 1 > 0:
                if val_loss_values[-1] < min_loss:
                    min_loss = val_loss_values[-1]
                    best_epoch = epoch
                    torch.save(model.state_dict(), './model_checkpoints/{}.pth'.format(config_str))
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt == args.patience:
                    break

                files = glob.glob('*.pth')
                for f in files:
                    epoch_nb = int(f.split('.')[0])
                    if epoch_nb < best_epoch:
                        os.remove(f)

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(f)
        print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - start))

        return best_epoch


def validate(datasets, funcs):
    model.eval()
    main_index = 0
    loss_sum = 0
    with torch.no_grad():
        pred = []
        gt = []
        batches = datasets.create_batches(funcs, datasets.collate)
        for index, batch_pair in enumerate(batches):
            data = datasets.transform(batch_pair)
            masks = create_batch_mask(data, device)
            
            prediction = model(data, masks, test = True)
            loss = F.binary_cross_entropy(prediction, data['target'], reduction='sum')
            main_index = main_index + len(batch_pair[2])
            loss_sum = loss_sum + loss.item()
            
            batch_gt = batch_pair[2]
            batch_pred = list(prediction.detach().cpu().numpy())

            pred = pred + batch_pred
            gt = gt + batch_gt
        
        loss = loss_sum / main_index
        gt = np.array(gt, dtype=np.float32)
        pred = np.array(pred, dtype=np.float32)
        score = roc_auc_score(gt, pred)

        return loss, score


def get_stats(array):
    
    mean = np.mean(np.asarray(array))
    std = np.std(np.asarray(array))

    return mean, std


def write_summary(args, config_str, stats):
    
    f = open("results/{}/{}.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("Loss : {:.4f} || AUC : {:.4f} ".format(stats[0], stats[1]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def write_summary_total(args, config_str, stats):
    
    f = open("results/{}/{}_total.txt".format(args.dataset, args.embedder), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("Loss : {:.4f}({:.4f}) || AUC : {:.4f}({:.4f}) ".format(stats[0], stats[1], stats[2], stats[3]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


if __name__ == "__main__":
    
    test_losses, test_aucs = [], []

    for i in range(5):
        best_model = train()
        model.load_state_dict(torch.load('./model_checkpoints/{}.pth'.format(config_str)))
        print('\nModel evaluation.')
        test_loss, test_auc = validate(dataset, dataset.testing_funcs)
        print('Test set results, loss = {:.6f}, AUC = {:.6f}'.format(test_loss, test_auc))

        test_losses.append(test_loss)
        test_aucs.append(test_auc)

        write_summary(args, config_str, [test_loss, test_auc])

    loss_mean, loss_std = get_stats(test_losses)
    auc_mean, auc_std = get_stats(test_aucs)

    write_summary_total(args, config_str, [loss_mean, loss_std, auc_mean, auc_std])