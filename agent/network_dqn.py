import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv


class QNetwork(nn.Module):
    def __init__(self, meta_data,
                       state_size,
                       num_nodes,
                       embed_dim,
                       num_heads,
                       num_HGT_layers,
                       num_q_layers,
                       use_gnn=True):
        super(QNetwork, self).__init__()
        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_q_layers = num_q_layers
        self.use_gnn = use_gnn

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

        self.q_head = nn.ModuleList()
        for i in range(num_q_layers):
            if i == 0:
                self.q_head.append(nn.Linear(embed_dim * 2, embed_dim * 2))
            elif i < num_q_layers - 1:
                self.q_head.append(nn.Linear(embed_dim * 2, embed_dim * 2))
            else:
                self.q_head.append(nn.Linear(embed_dim * 2, 1))

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

    def q_values(self, state, crane_id):
        x_dict, edge_index_dict = state.x_dict, state.edge_index_dict
        h_cranes, h_piles = self._encode(x_dict, edge_index_dict)

        h_piles_padding = h_piles.unsqueeze(-2).expand(-1, self.num_nodes["crane"], -1)
        h_cranes_padding = h_cranes.unsqueeze(-3).expand_as(h_piles_padding)
        h_actions = torch.cat((h_cranes_padding, h_piles_padding), dim=-1)

        for i in range(self.num_q_layers):
            if i < len(self.q_head) - 1:
                h_actions = self.q_head[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                q = self.q_head[i](h_actions).flatten()

        return q

    def act(self, state, mask, crane_id, epsilon=0.0):
        q = self.q_values(state, crane_id)
        mask = mask.transpose(0, 1).flatten()
        q_masked = q.clone()
        q_masked[~mask] = float('-inf')

        if torch.rand(1).item() < epsilon:
            valid_actions = mask.nonzero(as_tuple=True)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            action = torch.argmax(q_masked).item()

        return action, q_masked[action].item()

    def evaluate(self, batch_state, batch_action, batch_mask, crane_id):
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

        h_piles_padding = h_piles.unsqueeze(-2).expand(-1, -1, self.num_nodes["crane"], -1)
        h_cranes_padding = h_cranes.unsqueeze(-3).expand_as(h_piles_padding)
        h_actions = torch.cat((h_cranes_padding, h_piles_padding), dim=-1)

        for i in range(self.num_q_layers):
            if i < len(self.q_head) - 1:
                h_actions = self.q_head[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                batch_q = self.q_head[i](h_actions).flatten(1)

        batch_mask = batch_mask.transpose(1, 2).flatten(1)
        batch_q_masked = batch_q.clone()
        batch_q_masked[~batch_mask] = float('-inf')

        batch_q_taken = batch_q.gather(1, batch_action)
        batch_q_max_next = batch_q_masked.max(dim=1, keepdim=True)[0]

        return batch_q_taken, batch_q_max_next
