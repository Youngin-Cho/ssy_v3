import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import HGTConv, Linear, global_add_pool

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')


class Network(nn.Module):
    def __init__(self, state_size, action_size, meta_data, look_ahead, N):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.meta_data = meta_data
        self.look_ahead = look_ahead

        self.N = N
        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(1, self.n_cos + 1)]).view(1, 1, self.n_cos).to(device)
        layer = Linear

        self.conv1 = HGTConv(88, 512, meta_data, head=4)
        self.conv2 = HGTConv(512, 512, meta_data, head=4)
        self.cos_embedding = nn.Linear(self.n_cos, 512)
        self.ff_1 = layer(512, 512)
        self.ff_2 = layer(512, action_size)

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(device)  # (batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(self, input, num_tau=8):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """
        batch_size = input.num_graphs
        x_dict, edge_index_dict = input.x_dict, input.edge_index_dict

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.selu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.selu(x) for key, x in x_dict.items()}

        batch_idx = torch.arange(batch_size).to(device)
        batch_idx = batch_idx.repeat_interleave(int(x_dict["pile"].size(0) / batch_size))
        x = x_dict["crane"][:, 0:512] + global_add_pool(x_dict["pile"], batch_idx)

        cos, taus = self.calc_cos(batch_size, num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.selu(self.cos_embedding(cos)).view(batch_size, num_tau, 512)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, 512)

        x = torch.selu(self.ff_1(x))
        out = self.ff_2(x)

        return out.view(batch_size, num_tau, self.action_size), taus

    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.N)
        actions = quantiles.mean(dim=1)
        return actions