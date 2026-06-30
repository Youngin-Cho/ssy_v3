import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv


class QuantileNetwork(nn.Module):
    def __init__(self, meta_data,
                       state_size,
                       num_nodes,
                       embed_dim,
                       num_heads,
                       num_HGT_layers,
                       num_q_layers,
                       n_cos=64,
                       use_gnn=True):
        super(QuantileNetwork, self).__init__()
        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_q_layers = num_q_layers
        self.n_cos = n_cos
        self.use_gnn = use_gnn
        self.feature_dim = embed_dim * 2

        if use_gnn:
            self.conv = nn.ModuleList()
            for i in range(self.num_HGT_layers):
                if i == 0:
                    self.conv.append(HGTConv(self.state_size, embed_dim, meta_data, heads=num_heads))
                else:
                    self.conv.append(HGTConv(embed_dim, embed_dim, meta_data, heads=num_heads))
        else:
            self.mlp_crane = nn.ModuleList()
            self.mlp_pile = nn.ModuleList()
            for i in range(self.num_HGT_layers):
                if i == 0:
                    self.mlp_crane.append(nn.Linear(self.state_size["crane"], embed_dim))
                    self.mlp_pile.append(nn.Linear(self.state_size["pile"], embed_dim))
                else:
                    self.mlp_crane.append(nn.Linear(embed_dim, embed_dim))
                    self.mlp_pile.append(nn.Linear(embed_dim, embed_dim))

        # state-action feature psi(s, a) : (num_q_layers - 1)개의 hidden layer를 거쳐 feature_dim 벡터를 만든다
        self.feature_head = nn.ModuleList()
        for i in range(num_q_layers - 1):
            self.feature_head.append(nn.Linear(self.feature_dim, self.feature_dim))

        # quantile fraction tau를 cosine basis로 임베딩하는 layer (IQN 핵심 구성요소)
        self.quantile_embed = nn.Linear(n_cos, self.feature_dim)
        self.output_head = nn.Linear(self.feature_dim, 1)

    def _encode(self, x_dict, edge_index_dict):
        if self.use_gnn:
            for i in range(self.num_HGT_layers):
                x_dict = self.conv[i](x_dict, edge_index_dict)
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}
            h_cranes = x_dict["crane"]
            h_piles = x_dict["pile"]
        else:
            h_cranes = x_dict["crane"]
            h_piles = x_dict["pile"]
            for i in range(self.num_HGT_layers):
                h_cranes = self.mlp_crane[i](h_cranes)
                h_cranes = F.elu(h_cranes)
                h_piles = self.mlp_pile[i](h_piles)
                h_piles = F.elu(h_piles)
        return h_cranes, h_piles

    def _state_action_feature(self, h_cranes, h_piles, batched):
        if batched:
            h_piles_padding = h_piles.unsqueeze(-2).expand(-1, -1, self.num_nodes["crane"], -1)
            h_cranes_padding = h_cranes.unsqueeze(-3).expand_as(h_piles_padding)
            h_actions = torch.cat((h_cranes_padding, h_piles_padding), dim=-1)
            h_actions = h_actions.flatten(1, 2)  # (batch, action_dim, feature_dim)
        else:
            h_piles_padding = h_piles.unsqueeze(-2).expand(-1, self.num_nodes["crane"], -1)
            h_cranes_padding = h_cranes.unsqueeze(-3).expand_as(h_piles_padding)
            h_actions = torch.cat((h_cranes_padding, h_piles_padding), dim=-1)
            h_actions = h_actions.flatten(0, 1)  # (action_dim, feature_dim)

        psi = h_actions
        for layer in self.feature_head:
            psi = F.elu(layer(psi))

        return psi

    def _quantile_embedding(self, taus):
        i_pi = math.pi * torch.arange(1, self.n_cos + 1, device=taus.device, dtype=taus.dtype)
        cos_features = torch.cos(taus.unsqueeze(-1) * i_pi)  # (..., N, n_cos)
        phi = F.relu(self.quantile_embed(cos_features))  # (..., N, feature_dim)
        return phi

    def get_quantiles(self, psi, num_quantiles):
        batch_shape = psi.shape[:-2]
        taus = torch.rand(*batch_shape, num_quantiles, device=psi.device, dtype=psi.dtype)
        phi = self._quantile_embedding(taus)  # (..., N, feature_dim)

        combined = psi.unsqueeze(-2) * phi.unsqueeze(-3)  # (..., action_dim, N, feature_dim)
        quantiles = self.output_head(combined).squeeze(-1)  # (..., action_dim, N)

        return quantiles, taus

    def act(self, state, mask, crane_id, num_quantiles, epsilon=0.0):
        x_dict, edge_index_dict = state.x_dict, state.edge_index_dict
        h_cranes, h_piles = self._encode(x_dict, edge_index_dict)
        psi = self._state_action_feature(h_cranes, h_piles, batched=False)
        quantiles, _ = self.get_quantiles(psi, num_quantiles)  # (action_dim, N)
        q = quantiles.mean(dim=-1)

        mask = mask.transpose(0, 1).flatten()
        q_masked = q.clone()
        q_masked[~mask] = float('-inf')

        if torch.rand(1).item() < epsilon:
            valid_actions = mask.nonzero(as_tuple=True)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            action = torch.argmax(q_masked).item()

        return action, q_masked[action].item()

    def evaluate(self, batch_state, num_quantiles):
        batch_size = batch_state.num_graphs
        x_dict, edge_index_dict = batch_state.x_dict, batch_state.edge_index_dict

        if self.use_gnn:
            for i in range(self.num_HGT_layers):
                x_dict = self.conv[i](x_dict, edge_index_dict)
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}
            h_cranes = x_dict["crane"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_piles = x_dict["pile"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        else:
            h_cranes = batch_state["crane"]['x']
            h_piles = batch_state["pile"]['x']
            for i in range(self.num_HGT_layers):
                h_cranes = self.mlp_crane[i](h_cranes)
                h_cranes = F.elu(h_cranes)
                h_piles = self.mlp_pile[i](h_piles)
                h_piles = F.elu(h_piles)
            h_cranes = h_cranes.unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_piles = h_piles.unsqueeze(0).reshape(batch_size, -1, self.embed_dim)

        psi = self._state_action_feature(h_cranes, h_piles, batched=True)
        quantiles, taus = self.get_quantiles(psi, num_quantiles)  # (batch, action_dim, N)

        return quantiles, taus
