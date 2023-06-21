import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import HGTConv, Linear, RGATConv, global_add_pool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')


class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        # reset parameter as initialization of the layer
        self.reset_parameter()

    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input, noisy=True):
        # sample random noise in sigma weight buffer and bias buffer
        if noisy:
            self.epsilon_weight.normal_()
            weight = self.weight + self.sigma_weight * self.epsilon_weight
            bias = self.bias
            if bias is not None:
                self.epsilon_bias.normal_()
                bias = bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.weight
            bias = self.bias
        return F.linear(input, weight, bias)


class Network(nn.Module):
    def __init__(self, state_size, action_size, meta_data, look_ahead, n_units, N):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.meta_data = meta_data
        self.look_ahead = look_ahead

        self.n_units = n_units
        self.N = N
        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi * i for i in range(1, self.n_cos + 1)]).view(1, 1, self.n_cos).to(device)

        self.conv1 = HGTConv(self.state_size, n_units, meta_data, head=4)
        self.conv2 = HGTConv(n_units, n_units, meta_data, head=4)
        self.cos_embedding = nn.Linear(self.n_cos, n_units)
        self.ff_1 = NoisyLinear(n_units, n_units)
        self.ff_2 = NoisyLinear(n_units, n_units)
        self.advantage = NoisyLinear(n_units, action_size)
        self.value = NoisyLinear(n_units, 1)

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(device)  # (batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(self, input, num_tau=8, noisy=True, crane_id=0):
        batch_size = input.num_graphs
        x_dict, edge_index_dict = input.x_dict, input.edge_index_dict

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.selu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.selu(x) for key, x in x_dict.items()}

        batch_idx = torch.arange(batch_size).to(device)
        batch_idx_plate = batch_idx.repeat_interleave(int(x_dict["plate"].size(0) / batch_size))
        batch_idx_crane = batch_idx.repeat_interleave(int(x_dict["crane"].size(0) / batch_size))
        x = global_add_pool(x_dict["crane"], batch_idx_crane) + global_add_pool(x_dict["plate"], batch_idx_plate)

        cos, taus = self.calc_cos(batch_size, num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.n_units)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.n_units)

        x = torch.selu(self.ff_1(x, noisy=noisy))
        x = torch.selu(self.ff_2(x, noisy=noisy))
        advantage = self.advantage(x, noisy=noisy)
        value = self.value(x, noisy=noisy)
        out = value + advantage - advantage.mean(dim=1, keepdim=True)

        return out.view(batch_size, num_tau, self.action_size), taus

    def get_qvalues(self, inputs, noisy=True, crane_id=0):
        quantiles, _ = self.forward(inputs, self.N, noisy=noisy, crane_id=crane_id)
        actions = quantiles.mean(dim=1)
        return actions