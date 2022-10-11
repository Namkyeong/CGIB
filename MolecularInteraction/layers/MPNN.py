import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, Set2Set


class GatherModel(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 42.
    edge_input_dim : int
        Dimension of input edge feature, default to be 10.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 42.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    """

    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=3,
                 dropout = 0.0,
                 ):
        super(GatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim, 2, 1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_channels=node_hidden_dim,
                           out_channels=node_hidden_dim,
                           nn=edge_network,
                           aggr='add',
                           root_weight=True
                           )
        self.dropout = dropout

    def forward(self, g):
        """Returns the node embeddings after message passing phase.
        Parameters
        ----------
        g : Torch geometric batch data
            Input batch data for molecule(s)
        Returns
        -------
        res : node features
        """

        init = g.x.clone()
        out = F.relu(self.lin0(g.x))
        for i in range(self.num_step_message_passing):
            if len(g.edge_attr) != 0:
                m = torch.relu(self.conv(out, g.edge_index, g.edge_attr))
            else:
                m = torch.relu(self.conv.bias + out)
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init