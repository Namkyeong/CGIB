import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from torch_geometric.nn import Set2Set

from embedder import embedder
from layers import GINE
from utils import create_batch_mask

from torch_scatter import scatter_mean, scatter_add, scatter_std

import time


class CGIB_ModelTrainer(embedder):
    def __init__(self, args, train_df, valid_df, test_df, repeat, fold):
        embedder.__init__(self, args, train_df, valid_df, test_df, repeat, fold)

        self.model = CGIB(device = self.device, num_step_message_passing = self.args.message_passing).to(self.device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.args.patience, mode='max', verbose=True)
        
    def train(self):        

        loss_function_BCE = nn.BCEWithLogitsLoss(reduction='none')

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0
            preserve = 0

            start = time.time()

            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)
                
                outputs, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)])
                loss = loss_function_BCE(outputs, samples[2].reshape(-1, 1).to(self.device).float()).mean()

                # Information Bottleneck
                outputs, KL_Loss, solvent_pred_loss, preserve_rate = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], bottleneck = True)
                loss += loss_function_BCE(outputs, samples[2].reshape(-1, 1).to(self.device).float()).mean()
                loss += self.args.beta * KL_Loss
                loss += self.args.beta * solvent_pred_loss

                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                preserve += preserve_rate
            
            self.epoch_time = time.time() - start

            self.model.eval()
            self.evaluate(epoch)

            self.scheduler.step(self.val_roc_score)

            # Write Statistics
            self.writer.add_scalar("stats/preservation", preserve/bc, epoch)

            # Early stopping
            if len(self.best_val_rocs) > int(self.args.es / self.args.eval_freq):
                if self.best_val_rocs[-1] == self.best_val_rocs[-int(self.args.es / self.args.eval_freq)]:
                    if self.best_val_accs[-1] == self.best_val_accs[-int(self.args.es / self.args.eval_freq)]:
                        self.is_early_stop = True
                        break

        self.evaluate(epoch, final = True)
        self.writer.close()
        
        return self.best_test_roc, self.best_test_ap, self.best_test_f1, self.best_test_acc


class CGIB(nn.Module):
    """
    This the main class for CIGIN model
    """

    def __init__(self,
                device,
                node_input_dim=133,
                edge_input_dim=14,
                node_hidden_dim=300,
                edge_hidden_dim=300,
                num_step_message_passing=3,
                interaction='dot',
                num_step_set2_set=2,
                num_layer_set2set=1,
                ):
        super(CGIB, self).__init__()

        self.device = device

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction

        self.gather = GINE(self.node_input_dim, self.edge_input_dim, 
                            self.node_hidden_dim, self.num_step_message_passing,
                            )

        self.predictor = nn.Linear(8 * self.node_hidden_dim, 1)

        self.compressor = nn.Sequential(
            nn.Linear(2 * self.node_hidden_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            )
        
        self.solvent_predictor = nn.Linear(4 * self.node_hidden_dim, 4 * self.node_hidden_dim)
        
        self.mse_loss = torch.nn.MSELoss()

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def compress(self, solute_features):
        
        p = self.compressor(solute_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p
    
    def forward(self, data, bottleneck = False, test = False):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        # node embeddings after interaction phase
        solute_features = self.gather(solute)
        solvent_features = self.gather(solvent)

        # Add normalization
        self.solute_features = F.normalize(solute_features, dim = 1)
        self.solvent_features = F.normalize(solvent_features, dim = 1)

        # Interaction phase
        len_map = torch.sparse.mm(solute_len.t(), solvent_len)

        interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
        ret_interaction_map = torch.clone(interaction_map)
        ret_interaction_map = interaction_map * len_map.to_dense()
        interaction_map = interaction_map * len_map.to_dense()

        self.solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
        self.solute_prime = torch.mm(interaction_map, self.solvent_features)

        # Prediction phase
        self.solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        self.solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)

        if test:

            _, self.importance = self.compress(self.solute_features)
            self.importance = torch.sigmoid(self.importance)

        if bottleneck:

            lambda_pos, p = self.compress(self.solute_features)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos

            # Get Stats
            preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

            static_solute_feature = self.solute_features.clone().detach()
            node_feature_mean = scatter_mean(static_solute_feature, solute.batch, dim = 0)[solute.batch]
            node_feature_std = scatter_std(static_solute_feature, solute.batch, dim = 0)[solute.batch]
            # node_feature_std, node_feature_mean = torch.std_mean(static_solute_feature, dim=0)
            
            noisy_node_feature_mean = lambda_pos * self.solute_features + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std

            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
            noisy_solute_subgraphs = self.set2set(noisy_node_feature, solute.batch)

            epsilon = 1e-7

            KL_tensor = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim = 1), solute.batch).reshape(-1, 1) + \
                        scatter_add((((noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon)) ** 2), solute.batch, dim = 0)
            KL_Loss = torch.mean(KL_tensor)
            
            # Predict Solvent
            self.solvent_features_s2s = self.set2set(self.solvent_features, solvent.batch)
            solvent_pred_loss = self.mse_loss(self.solvent_features_s2s, self.solvent_predictor(noisy_solute_subgraphs))

            # Prediction Y
            final_features = torch.cat((noisy_solute_subgraphs, self.solvent_features_s2s), 1)
            predictions = self.predictor(final_features)

            return predictions, KL_Loss, solvent_pred_loss, preserve_rate
        
        else:

            self.solute_features_s2s = self.set2set(self.solute_features, solute.batch)
            self.solvent_features_s2s = self.set2set(self.solvent_features, solvent.batch)

            final_features = torch.cat((self.solute_features_s2s, self.solvent_features_s2s), 1)
            predictions = self.predictor(final_features)

            if test:
                return torch.sigmoid(predictions), ret_interaction_map

            else:
                return predictions, ret_interaction_map