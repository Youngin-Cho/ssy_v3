import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv
from torch.distributions import Categorical


class Scheduler(nn.Module):
    def __init__(self, meta_data,
                       state_size,
                       num_nodes,
                       embed_dim,
                       num_heads,
                       num_HGT_layers,
                       num_actor_layers,
                       num_critic_layers,
                       use_gnn=True):
        super(Scheduler, self).__init__()
        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_actor_layers = num_actor_layers
        self.num_critic_layers = num_critic_layers
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

        self.actor = nn.ModuleList()
        for i in range(num_actor_layers):
            if i == 0:
                self.actor.append(nn.Linear(embed_dim * 2, embed_dim * 2))
            elif 0 < i < num_actor_layers - 1:
                self.actor.append(nn.Linear(embed_dim * 2, embed_dim * 2))
            else:
                self.actor.append(nn.Linear(embed_dim * 2, 1))

        self.critic = nn.ModuleList()
        for i in range(num_critic_layers):
            if i == 0:
                self.critic.append(nn.Linear(embed_dim * 2, embed_dim * 2))
            elif i < num_critic_layers - 1:
                self.critic.append(nn.Linear(embed_dim * 2, embed_dim * 2))
            else:
                self.critic.append(nn.Linear(embed_dim * 2, 1))

    def act(self, state, mask, crane_id, greedy=False):
        x_dict, edge_index_dict = state.x_dict, state.edge_index_dict

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

        h_cranes_pooled = h_cranes.mean(dim=-2)
        h_piles_pooled = h_piles.mean(dim=-2)

        h_piles_padding = h_piles.unsqueeze(-2).expand(-1, self.num_nodes["crane"], -1)
        h_cranes_padding = h_cranes.unsqueeze(-3).expand_as(h_piles_padding)

        # h_cranes_pooled_padding = h_cranes_pooled[None, None, :].expand_as(h_cranes_padding)
        # h_piles_pooled_padding = h_piles_pooled[None, None, :].expand_as(h_piles_padding)

        # h_actions = torch.cat((h_cranes_padding, h_piles_padding,
        #                        h_cranes_pooled_padding, h_piles_pooled_padding), dim=-1)
        h_actions = torch.cat((h_cranes_padding, h_piles_padding), dim=-1)
        h_pooled = torch.cat((h_cranes_pooled, h_piles_pooled), dim=-1)

        for i in range(self.num_actor_layers):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                logits = self.actor[i](h_actions).flatten()

        mask = mask.transpose(0, 1).flatten()
        logits[~mask] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        if greedy:
            action = torch.argmax(probs)
            action_logprob = dist.log_prob(action)
        else:
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            while action_logprob < -15:
                action = dist.sample()
                action_logprob = dist.log_prob(action)

        for i in range(self.num_critic_layers):
            if i < len(self.critic) - 1:
                h_pooled = self.critic[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                state_value = self.critic[i](h_pooled)

        return action.item(), action_logprob.item(), state_value.squeeze().item()

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

        h_cranes_pooled = h_cranes.mean(dim=-2)
        h_piles_pooled = h_piles.mean(dim=-2)

        h_piles_padding = h_piles.unsqueeze(-2).expand(-1, -1, self.num_nodes["crane"], -1)
        h_cranes_padding = h_cranes.unsqueeze(-3).expand_as(h_piles_padding)

        # h_cranes_pooled_padding = h_cranes_pooled[:, None, None, :].expand_as(h_cranes_padding)
        # h_piles_pooled_padding = h_piles_pooled[:, None, None, :].expand_as(h_piles_padding)

        # h_actions = torch.cat((h_cranes_padding, h_piles_padding,
        #                        h_cranes_pooled_padding, h_piles_pooled_padding), dim=-1)
        h_actions = torch.cat((h_cranes_padding, h_piles_padding), dim=-1)
        h_pooled = torch.cat((h_cranes_pooled, h_piles_pooled), dim=-1)

        for i in range(self.num_actor_layers):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                batch_logits = self.actor[i](h_actions).flatten(1)

        batch_mask = batch_mask.transpose(1, 2).flatten(1)
        batch_logits[~batch_mask] = float('-inf')
        batch_probs = F.softmax(batch_logits, dim=1)
        batch_dist = Categorical(batch_probs)
        batch_action_logprobs = batch_dist.log_prob(batch_action.squeeze()).unsqueeze(-1)

        for i in range(self.num_critic_layers):
            if i < len(self.critic) - 1:
                h_pooled = self.critic[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                batch_state_values = self.critic[i](h_pooled)

        batch_dist_entropys = batch_dist.entropy().unsqueeze(-1)

        return batch_action_logprobs, batch_state_values, batch_dist_entropys