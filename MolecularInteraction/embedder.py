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
from utils import create_batch_mask
from data import Dataclass
from torch_geometric.data import DataLoader


class embedder:

    def __init__(self, args, train_df, valid_df, test_df, repeat, fold):
        self.args = args
        self.config_str = "experiment{}_fold{}_".format(repeat + 1, fold + 1) + config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
        if args.writer:
            if repeat == 0:
                self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))
            else:
                self.writer = SummaryWriter(log_dir="runs_/{}".format(self.config_str))
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

        self.best_val_loss = 100000000.0
        self.best_val_losses = []

        self.fold = fold
    

    def evaluate(self, epoch, final = False):
        
        valid_losses = []
        valid_mae_losses = []
    
        test_losses = []
        test_mae_losses = []

        for bc, samples in enumerate(self.val_loader):

            masks = create_batch_mask(samples)
            output, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], test = True)
            
            val_loss = ((output - samples[2].reshape(-1, 1).to(self.device))**2).reshape(-1)
            val_mae_loss = torch.abs((output - samples[2].reshape(-1, 1).to(self.device))).reshape(-1)
            valid_losses.append(val_loss.cpu().detach().numpy())
            valid_mae_losses.append(val_mae_loss.cpu().detach().numpy())        


        for bc, samples in enumerate(self.test_loader):
            
            masks = create_batch_mask(samples)
            output, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], test = True)
            
            test_loss = ((output - samples[2].reshape(-1, 1).to(self.device))**2).reshape(-1)
            test_mae_loss = torch.abs((output - samples[2].reshape(-1, 1).to(self.device))).reshape(-1)
            test_losses.append(test_loss.cpu().detach().numpy())
            test_mae_losses.append(test_mae_loss.cpu().detach().numpy())

                    
        self.val_loss = np.mean(np.hstack(valid_losses))
        self.val_rmse_loss = np.sqrt(self.val_loss)
        self.val_mae_loss = np.mean(np.hstack(valid_mae_losses))
        self.test_loss = np.mean(np.hstack(test_losses))
        self.test_rmse_loss = np.sqrt(self.test_loss)
        self.test_mae_loss = np.mean(np.hstack(test_mae_losses))

        self.writer.add_scalar("accs/train_loss", self.train_loss, epoch)
        self.writer.add_scalar("accs/valid_RMSE", self.val_rmse_loss, epoch)
        self.writer.add_scalar("accs/test_RMSE", self.test_rmse_loss, epoch)

        if self.val_loss < self.best_val_loss :
            self.best_val_loss = self.val_loss
            self.best_test_loss = self.test_loss
            self.best_val_mae_loss = self.val_mae_loss
            self.best_test_mae_loss = self.test_mae_loss
            self.best_val_rmse_loss = self.val_rmse_loss
            self.best_test_rmse_loss = self.test_rmse_loss
            self.best_epoch = epoch

        self.best_val_losses.append(self.best_val_loss)

        self.eval_config = "[Epoch: {}] valid MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f} || test MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f}".format(epoch, self.val_loss, self.val_rmse_loss, self.val_mae_loss, self.test_loss, self.test_rmse_loss, self.test_mae_loss)
        self.best_config = "[Best Epoch: {}] Best valid MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f} || Best test MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f}".format(self.best_epoch, self.best_val_loss, self.best_val_rmse_loss, self.best_val_mae_loss, self.best_test_loss, self.best_test_rmse_loss, self.best_test_mae_loss)

        print(self.eval_config)
        print(self.best_config)
