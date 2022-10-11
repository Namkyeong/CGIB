import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GINEConv(MessagePassing):
    def __init__(self, edge_dim, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINEConv, self).__init__(aggr="add")

        # self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), 
        #         torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.mlp = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), 
                                    nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = nn.Linear(edge_dim, emb_dim)

    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)

        self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + 
                        self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        
        return aggr_out


class GINE(nn.Module):

    def __init__(self, node_input_dim, edge_input_dim, hidden_channels, num_layers):

        super(GINE, self).__init__()

        self.node_encoder = nn.Linear(node_input_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.num_layers = num_layers

        for i in range(self.num_layers):
            self.convs.append(GINEConv(edge_input_dim, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.out_lin = nn.Linear(hidden_channels * (num_layers + 1), hidden_channels)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()

    def forward(self, batch_data):

        h = self.node_encoder(batch_data.x)

        h_list = [h]

        for i, conv in enumerate(self.convs):
            h = conv(h_list[i], batch_data.edge_index, batch_data.edge_attr)
            # h = self.bns[i](h)
            h = F.relu(h)
            h_list.append(h)
            
        h_list = torch.cat(h_list, dim = 1)
        h_list = self.out_lin(h_list)

        return h_list