import numpy as np
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)

from tensorboardX import SummaryWriter
import os

from argument import config2string
from utils import create_batch_mask, get_roc_score
from data import Dataclass
from torch_geometric.data import DataLoader


class embedder:

    def __init__(self, args, train_df, valid_df, test_df, repeat, fold):
        self.args = args
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
        if args.writer:
            self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))
        else:
            self.writer = SummaryWriter(log_dir="runs_/{}".format(self.config_str))

        # Model Checkpoint Path
        CHECKPOINT_PATH = "model_checkpoints/{}/".format(args.embedder)
        self.check_dir = CHECKPOINT_PATH + self.config_str + ".pt"

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        self.train_dataset = Dataclass(train_df)
        self.val_dataset = Dataclass(valid_df)
        self.test_dataset = Dataclass(test_df)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size = args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size = args.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size = args.batch_size)

        self.is_early_stop = False

        self.best_val_roc = -1.0
        self.best_val_rocs = []

        self.best_val_acc = -1.0
        self.best_val_accs = []

    def evaluate(self, epoch, final = False):
        
        valid_outputs, valid_labels = [], []
        test_outputs, test_labels = [], []

        for bc, samples in enumerate(self.val_loader):

            masks = create_batch_mask(samples)
            output, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], test = True)
            
            valid_outputs.append(output.reshape(-1).detach().cpu().numpy())
            valid_labels.append(samples[2].reshape(-1).detach().cpu().numpy())
        
        valid_outputs = np.hstack(valid_outputs)
        valid_labels = np.hstack(valid_labels)

        self.val_roc_score, val_ap_score, val_f1_score, self.val_acc_score = get_roc_score(valid_outputs, valid_labels)

        self.writer.add_scalar("val/val_roc_score", self.val_roc_score, epoch)
        self.writer.add_scalar("val/val_ap_score", val_ap_score, epoch)
        self.writer.add_scalar("val/val_f1_score", val_f1_score, epoch)
        self.writer.add_scalar("val/val_acc_score", self.val_acc_score, epoch)


        for bc, samples in enumerate(self.test_loader):
                
            masks = create_batch_mask(samples)
            output, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], test = True)
            
            test_outputs.append(output.reshape(-1).detach().cpu().numpy())
            test_labels.append(samples[2].reshape(-1).detach().cpu().numpy())

        test_outputs = np.hstack(test_outputs)
        test_labels = np.hstack(test_labels)

        test_roc_score, test_ap_score, test_f1_score, test_acc_score = get_roc_score(test_outputs, test_labels)

        self.writer.add_scalar("test/test_roc_score", test_roc_score, epoch)
        self.writer.add_scalar("test/test_ap_score", test_ap_score, epoch)
        self.writer.add_scalar("test/test_f1_score", test_f1_score, epoch)
        self.writer.add_scalar("test/test_acc_score", test_acc_score, epoch)

        # Save ROC score
        if self.val_roc_score > self.best_val_roc :
            
            # Save validation score
            self.best_val_roc = self.val_roc_score
            self.best_val_ap = val_ap_score
            # Save test score
            self.best_test_roc = test_roc_score
            self.best_test_ap = test_ap_score

            # Save epoch
            self.best_roc_epoch = epoch
        
        # Save f1 score
        if self.val_acc_score > self.best_val_acc :

            self.best_val_f1 = val_f1_score
            self.best_val_acc = self.val_acc_score
            
            self.best_test_f1 = test_f1_score
            self.best_test_acc = test_acc_score
            
            # Save epoch
            self.best_f1_epoch = epoch

        self.best_val_rocs.append(self.best_val_roc)
        self.best_val_accs.append(self.best_val_acc)

        self.eval_config = "[Epoch: {} ({:.4f} sec)] Valid ROC: {:.4f} / AP: {:.4f} / F1: {:.4f} / Acc: {:.4f} || Test ROC: {:.4f} / AP: {:.4f} / F1: {:.4f} / Acc: {:.4f} ".format(epoch, self.epoch_time, self.val_roc_score, val_ap_score, val_f1_score, self.val_acc_score, test_roc_score, test_ap_score, test_f1_score, test_acc_score)
        self.best_config_roc = "[Best ROC Epoch: {}] Best Valid ROC: {:.4f} / AP: {:.4f} || Best Test ROC: {:.4f} / AP: {:.4f} ".format(self.best_roc_epoch, self.best_val_roc, self.best_val_ap, self.best_test_roc, self.best_test_ap)
        self.best_config_f1 = "[Best F1 Epoch: {}] Best Valid F1: {:.4f} / Acc: {:.4f} || Best Test F1: {:.4f} / Acc: {:.4f} ".format(self.best_f1_epoch, self.best_val_f1, self.best_val_acc, self.best_test_f1, self.best_test_acc)

        print(self.eval_config)
        print(self.best_config_roc)
        print(self.best_config_f1)
