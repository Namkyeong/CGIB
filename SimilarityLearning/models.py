import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv, Set2Set

from layers import ReadoutModule, MLPModule, CrossGraphConvolution, HyperedgeConv, HyperedgePool
from utils import hypergraph_construction
from torch_scatter import scatter_mean, scatter_add, scatter_std

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.k = args.k
        self.mode = args.mode

        self.num_features = args.num_features

        self.conv0 = GCNConv(self.num_features, self.nhid)
        self.conv1 = HypergraphConv(self.nhid, self.nhid)
        self.cross_conv1 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool1 = HyperedgePool(self.nhid, self.args.ratio1)
        
        self.conv2 = HyperedgeConv(self.nhid, self.nhid)
        self.cross_conv2 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool2 = HyperedgePool(self.nhid, self.args.ratio2)
        
        self.conv3 = HyperedgeConv(self.nhid, self.nhid)
        self.cross_conv3 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool3 = HyperedgePool(self.nhid, self.args.ratio3)

        self.readout0 = ReadoutModule(self.args)
        self.readout1 = ReadoutModule(self.args)
        self.readout2 = ReadoutModule(self.args)
        self.readout3 = ReadoutModule(self.args)

        self.mlp = MLPModule(self.args)

    def forward(self, data):
        edge_index_1 = data['g1'].edge_index
        edge_index_2 = data['g2'].edge_index
        
        edge_attr_1 = data['g1'].edge_attr
        edge_attr_2 = data['g2'].edge_attr

        features_1 = data['g1'].x
        features_2 = data['g2'].x

        batch_1 = data['g1'].batch
        batch_2 = data['g2'].batch
        
        # Layer 0
        # Graph Convolution Operation
        f1_conv0 = F.leaky_relu(self.conv0(features_1, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv0 = F.leaky_relu(self.conv0(features_2, edge_index_2, edge_attr_2), negative_slope=0.2)

        att_f1_conv0 = self.readout0(f1_conv0, batch_1)
        att_f2_conv0 = self.readout0(f2_conv0, batch_2)
        score0 = torch.cat([att_f1_conv0, att_f2_conv0], dim=1)

        edge_index_1, edge_attr_1 = hypergraph_construction(edge_index_1, edge_attr_1, num_nodes=features_1.size(0), k=self.k, mode=self.mode)
        edge_index_2, edge_attr_2 = hypergraph_construction(edge_index_2, edge_attr_2, num_nodes=features_2.size(0), k=self.k, mode=self.mode)
        
        # Layer 1
        # Hypergraph Convolution Operation
        f1_conv1 = F.leaky_relu(self.conv1(f1_conv0, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv1 = F.leaky_relu(self.conv1(f2_conv0, edge_index_2, edge_attr_2), negative_slope=0.2)
        
        # Hyperedge Pooling
        edge1_conv1, edge1_index_pool1, edge1_attr_pool1, edge1_batch_pool1 = self.pool1(f1_conv1, batch_1, edge_index_1, edge_attr_1)
        edge2_conv1, edge2_index_pool1, edge2_attr_pool1, edge2_batch_pool1 = self.pool1(f2_conv1, batch_2, edge_index_2, edge_attr_2)
        
        # Cross Graph Convolution
        hyperedge1_cross_conv1, hyperedge2_cross_conv1 = self.cross_conv1(edge1_conv1, edge1_batch_pool1, edge2_conv1, edge2_batch_pool1)

        # Readout Module
        att_f1_conv1 = self.readout1(hyperedge1_cross_conv1, edge1_batch_pool1)
        att_f2_conv1 = self.readout1(hyperedge2_cross_conv1, edge2_batch_pool1)
        score1 = torch.cat([att_f1_conv1, att_f2_conv1], dim=1)

        # Layer 2
        # Hypergraph Convolution Operation
        f1_conv2 = F.leaky_relu(self.conv2(hyperedge1_cross_conv1, edge1_index_pool1, edge1_attr_pool1), negative_slope=0.2)
        f2_conv2 = F.leaky_relu(self.conv2(hyperedge2_cross_conv1, edge2_index_pool1, edge2_attr_pool1), negative_slope=0.2)

        # Hyperedge Pooling
        edge1_conv2, edge1_index_pool2, edge1_attr_pool2, edge1_batch_pool2 = self.pool2(f1_conv2, edge1_batch_pool1, edge1_index_pool1, edge1_attr_pool1)
        edge2_conv2, edge2_index_pool2, edge2_attr_pool2, edge2_batch_pool2 = self.pool2(f2_conv2, edge2_batch_pool1, edge2_index_pool1, edge2_attr_pool1)
        
        # Cross Graph Convolution
        hyperedge1_cross_conv2, hyperedge2_cross_conv2 = self.cross_conv2(edge1_conv2, edge1_batch_pool2, edge2_conv2, edge2_batch_pool2)

        # Readout Module
        att_f1_conv2 = self.readout2(hyperedge1_cross_conv2, edge1_batch_pool2)
        att_f2_conv2 = self.readout2(hyperedge2_cross_conv2, edge2_batch_pool2)
        score2 = torch.cat([att_f1_conv2, att_f2_conv2], dim=1)

        # Layer 3
        # Hypergraph Convolution Operation
        f1_conv3 = F.leaky_relu(self.conv3(hyperedge1_cross_conv2, edge1_index_pool2, edge1_attr_pool2), negative_slope=0.2)
        f2_conv3 = F.leaky_relu(self.conv3(hyperedge2_cross_conv2, edge2_index_pool2, edge2_attr_pool2), negative_slope=0.2)

        # Hyperedge Pooling
        edge1_conv3, edge1_index_pool3, edge1_attr_pool3, edge1_batch_pool3 = self.pool3(f1_conv3, edge1_batch_pool2, edge1_index_pool2, edge1_attr_pool2)
        edge2_conv3, edge2_index_pool3, edge2_attr_pool3, edge2_batch_pool3 = self.pool3(f2_conv3, edge2_batch_pool2, edge2_index_pool2, edge2_attr_pool2)

        # Cross Graph Convolution
        hyperedge1_cross_conv3, hyperedge2_cross_conv3 = self.cross_conv3(edge1_conv3, edge1_batch_pool3, edge2_conv3, edge2_batch_pool3)

        # Readout Module
        att_f1_conv3 = self.readout3(hyperedge1_cross_conv3, edge1_batch_pool3)
        att_f2_conv3 = self.readout3(hyperedge2_cross_conv3, edge2_batch_pool3)
        score3 = torch.cat([att_f1_conv3, att_f2_conv3], dim=1)

        scores = torch.cat([score0, score1, score2, score3], dim=1)
        scores = self.mlp(scores)

        return scores


class CGIB(nn.Module):
    def __init__(self, args, device):
        super(CGIB, self).__init__()
        self.args = args
        self.nhid = args.nhid

        self.num_step_set2set = 2
        self.num_layer_set2set = 1

        self.device = device

        self.num_features = args.num_features

        self.conv0 = GCNConv(self.num_features, self.nhid)
        self.conv1 = GCNConv(self.nhid, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        self.set2set = Set2Set(6 * self.nhid, self.num_step_set2set, self.num_layer_set2set)
        
        self.predictor = nn.Sequential(
            nn.Linear(24 * self.nhid, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.compressor = nn.Sequential(
            nn.Linear(6 * self.nhid, self.nhid),
            nn.BatchNorm1d(self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, 1)
            )
        
        self.solvent_predictor = nn.Linear(12 * self.nhid, 12 * self.nhid)
        self.mse_loss = torch.nn.MSELoss()

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

    def forward(self, data, int_map, bottleneck = False, test = False):
        edge_index_1 = data['g1'].edge_index
        edge_index_2 = data['g2'].edge_index
        
        edge_attr_1 = data['g1'].edge_attr
        edge_attr_2 = data['g2'].edge_attr

        features_1 = data['g1'].x
        features_2 = data['g2'].x

        batch_1 = data['g1'].batch
        batch_2 = data['g2'].batch
        
        # Graph Encoding Process
        f1_conv0 = F.leaky_relu(self.conv0(features_1, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv0 = F.leaky_relu(self.conv0(features_2, edge_index_2, edge_attr_2), negative_slope=0.2)

        f1_conv1 = F.leaky_relu(self.conv1(f1_conv0, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv1 = F.leaky_relu(self.conv1(f2_conv0, edge_index_2, edge_attr_2), negative_slope=0.2)

        f1_conv2 = F.leaky_relu(self.conv2(f1_conv1, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv2 = F.leaky_relu(self.conv2(f2_conv1, edge_index_2, edge_attr_2), negative_slope=0.2)

        f1_conv0 = torch.cat([f1_conv0, f1_conv1, f1_conv2], dim = 1)
        f2_conv0 = torch.cat([f2_conv0, f2_conv1, f2_conv2], dim = 1)

        f1_conv0 = F.normalize(f1_conv0, dim = 1)
        f2_conv0 = F.normalize(f2_conv0, dim = 1)

        # Interaction phase
        len_map = torch.sparse.mm(int_map[0].t(), int_map[1])

        interaction_map = torch.mm(f1_conv0, f2_conv0.t())
        interaction_map = interaction_map * len_map.to_dense()

        f2_prime = torch.mm(interaction_map.t(), f1_conv0)
        f1_prime = torch.mm(interaction_map, f2_conv0)

        # Prediction phase
        f1_features = torch.cat((f1_conv0, f1_prime), dim=1)
        f2_features = torch.cat((f2_conv0, f2_prime), dim=1)

        if (bottleneck == True) and (test == False):

            lambda_pos, p = self.compress(f1_features)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos

            static_solute_feature = f1_features.clone().detach()
            node_feature_mean = scatter_mean(static_solute_feature, batch_1, dim = 0)[batch_1]
            node_feature_std = scatter_std(static_solute_feature, batch_1, dim = 0)[batch_1]

            noisy_node_feature_mean = lambda_pos * f1_features + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std

            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
            noisy_subgraphs1 = self.set2set(noisy_node_feature, batch_1)

            epsilon = 1e-7

            KL_tensor = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim = 1), batch_1).reshape(-1, 1) + \
                            scatter_add((((noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon)) ** 2), batch_1, dim = 0)
            KL_Loss = torch.mean(KL_tensor)

            graph2 = self.set2set(f2_features, batch_2)
            graph2_pred_loss = self.mse_loss(graph2, self.solvent_predictor(noisy_subgraphs1))

            final_features = torch.cat((noisy_subgraphs1, graph2), 1)
            scores = self.predictor(final_features)
            scores = torch.sigmoid(scores).view(-1)

            return scores, KL_Loss, graph2_pred_loss

        else:

            graph1 = self.set2set(f1_features, batch_1)
            graph2 = self.set2set(f2_features, batch_2)

            final_features = torch.cat((graph1, graph2), 1)
            scores = self.predictor(final_features)
            scores = torch.sigmoid(scores).view(-1)
        
            return scores


class CGIB_cont(nn.Module):
    def __init__(self, args, device):
        super(CGIB_cont, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.tau = args.tau

        self.num_step_set2set = 2
        self.num_layer_set2set = 1

        self.device = device

        self.num_features = args.num_features

        self.conv0 = GCNConv(self.num_features, self.nhid)
        self.conv1 = GCNConv(self.nhid, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        self.set2set = Set2Set(6 * self.nhid, self.num_step_set2set, self.num_layer_set2set)
        
        self.predictor = nn.Sequential(
            nn.Linear(24 * self.nhid, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.compressor = nn.Sequential(
            nn.Linear(6 * self.nhid, self.nhid),
            nn.BatchNorm1d(self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, 1)
            )
        
        self.solvent_predictor = nn.Linear(12 * self.nhid, 12 * self.nhid)
        self.mse_loss = torch.nn.MSELoss()

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

    def forward(self, data, int_map, bottleneck = False, test = False):
        edge_index_1 = data['g1'].edge_index
        edge_index_2 = data['g2'].edge_index
        
        edge_attr_1 = data['g1'].edge_attr
        edge_attr_2 = data['g2'].edge_attr

        features_1 = data['g1'].x
        features_2 = data['g2'].x

        batch_1 = data['g1'].batch
        batch_2 = data['g2'].batch
        
        # Graph Encoding Process
        f1_conv0 = F.leaky_relu(self.conv0(features_1, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv0 = F.leaky_relu(self.conv0(features_2, edge_index_2, edge_attr_2), negative_slope=0.2)

        f1_conv1 = F.leaky_relu(self.conv1(f1_conv0, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv1 = F.leaky_relu(self.conv1(f2_conv0, edge_index_2, edge_attr_2), negative_slope=0.2)

        f1_conv2 = F.leaky_relu(self.conv2(f1_conv1, edge_index_1, edge_attr_1), negative_slope=0.2)
        f2_conv2 = F.leaky_relu(self.conv2(f2_conv1, edge_index_2, edge_attr_2), negative_slope=0.2)

        f1_conv0 = torch.cat([f1_conv0, f1_conv1, f1_conv2], dim = 1)
        f2_conv0 = torch.cat([f2_conv0, f2_conv1, f2_conv2], dim = 1)

        f1_conv0 = F.normalize(f1_conv0, dim = 1)
        f2_conv0 = F.normalize(f2_conv0, dim = 1)

        # Interaction phase
        len_map = torch.sparse.mm(int_map[0].t(), int_map[1])

        interaction_map = torch.mm(f1_conv0, f2_conv0.t())
        interaction_map = interaction_map * len_map.to_dense()

        f2_prime = torch.mm(interaction_map.t(), f1_conv0)
        f1_prime = torch.mm(interaction_map, f2_conv0)

        # Prediction phase
        f1_features = torch.cat((f1_conv0, f1_prime), dim=1)
        f2_features = torch.cat((f2_conv0, f2_prime), dim=1)

        if (bottleneck == True) and (test == False):

            lambda_pos, p = self.compress(f1_features)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos

            static_solute_feature = f1_features.clone().detach()
            node_feature_mean = scatter_mean(static_solute_feature, batch_1, dim = 0)[batch_1]
            node_feature_std = scatter_std(static_solute_feature, batch_1, dim = 0)[batch_1]

            noisy_node_feature_mean = lambda_pos * f1_features + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std

            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
            noisy_subgraphs1 = self.set2set(noisy_node_feature, batch_1)

            epsilon = 1e-7

            KL_tensor = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim = 1), batch_1).reshape(-1, 1) + \
                            scatter_add((((noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon)) ** 2), batch_1, dim = 0)
            KL_Loss = torch.mean(KL_tensor)

            graph2 = self.set2set(f2_features, batch_2)
            cont_loss = self.contrastive_loss(noisy_subgraphs1, graph2, self.tau)

            final_features = torch.cat((noisy_subgraphs1, graph2), 1)
            scores = self.predictor(final_features)
            scores = torch.sigmoid(scores).view(-1)

            return scores, KL_Loss, cont_loss

        else:

            graph1 = self.set2set(f1_features, batch_1)
            graph2 = self.set2set(f2_features, batch_2)

            final_features = torch.cat((graph1, graph2), 1)
            scores = self.predictor(final_features)
            scores = torch.sigmoid(scores).view(-1)
        
            return scores

    def contrastive_loss(self, solute, solvent, tau):

        batch_size, _ = solute.size()
        solute_abs = solute.norm(dim = 1)
        solvent_abs = solvent.norm(dim = 1)        

        sim_matrix = torch.einsum('ik,jk->ij', solute, solvent) / torch.einsum('i,j->ij', solute_abs, solvent_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
