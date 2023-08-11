import torch
import random
import numpy as np
import pandas as pd

from torch_geometric.data import HeteroData, Data
from environment.simulation import Management


class SteelStockYard:
    def __init__(self, data_src, look_ahead=2, rows=("A", "B"), working_crane_ids=("Crane-1", "Crane-2"), safety_margin=5):

        self.data_src = data_src
        self.look_ahead = look_ahead
        self.rows = rows
        self.working_crane_ids = working_crane_ids
        self.safety_margin = safety_margin

        if type(self.data_src) is DataGenerator:
            self.df_storage, self.df_reshuffle, self.df_retrieval = self.data_src.generate()
        else:
            self.df_storage = pd.read_excel(self.data_src, sheet_name="storage", engine="openpyxl")
            self.df_reshuffle = pd.read_excel(self.data_src, sheet_name="reshuffle", engine="openpyxl")
            self.df_retrieval = pd.read_excel(self.data_src, sheet_name="retrieval", engine="openpyxl")

        self.action_size = 2 * 40 + 2 + 1
        self.state_size = {"crane": 2, "pile": 6}
        self.meta_data = (["crane", "pile"],
                          [("crane", "interfering", "crane"),
                           ("pile", "moving_rev", "crane"),
                           ("crane", "moving", "pile"),
                           ("pile", "stacking", "pile")])

        self.crane_list = ["Crane-1", "Crane-2"]
        self.pile_list = list(self.df_storage["pileno"].unique()) + list(self.df_reshuffle["pileno"].unique())
        self.model = Management(self.df_storage, self.df_reshuffle, self.df_retrieval,
                                rows=self.rows, working_crane_ids=working_crane_ids, safety_margin=safety_margin)

        self.action_mapping = {i + 1: pile_name for i, pile_name in enumerate(self.model.piles.keys())}
        self.action_mapping_inverse = {y: x for x, y in self.action_mapping.items()}
        self.crane_in_decision = None
        self.time = 0.0

    def step(self, action):
        done = False

        if action == 0:
            action = "Waiting"
        else:
            action = self.action_mapping[action]
        self.model.do_action.succeed(action)
        self.model.decision_time = False

        while True:
            if self.model.decision_time:
                break
            else:
                if len(self.model.move_list) == 0:
                    done = True
                    break
                else:
                    self.model.env.step()

        reward = self._calculate_reward()
        next_state = self._get_state()

        self.crane_in_decision = self.model.crane_in_decision
        info = {"crane_id": self.crane_in_decision}
        self.time = self.model.env.now

        return next_state, reward, done, info

    def reset(self):
        self.model = Management(self.df_storage, self.df_reshuffle, self.df_retrieval,
                                rows=self.rows, working_crane_ids=self.working_crane_ids, safety_margin=self.safety_margin)

        while True:
            if self.model.decision_time:
                break
            else:
                self.model.env.step()

        self.crane_in_decision = self.model.crane_in_decision
        info = {"crane_id": self.crane_in_decision}

        return self._get_state(), info

    def get_possible_actions(self):
        possbile_actions = []

        for from_pile_name in np.unique(self.model.move_list):
            to_pile_name = self.model.piles[from_pile_name].plates[-1].to_pile

            possible = True
            from_pile_x = self.model.piles[from_pile_name].coord[0]
            to_pile_x = self.model.piles[to_pile_name].coord[0]

            if self.crane_in_decision == 0 and from_pile_x > 44 - self.safety_margin:
                possible = False
            if self.crane_in_decision == 1 and from_pile_x < 1 + self.safety_margin:
                possible = False
            if self.crane_in_decision == 0 and to_pile_x > 44 - self.safety_margin:
                possible = False
            if self.crane_in_decision == 1 and to_pile_x < 1 + self.safety_margin:
                possible = False
            if from_pile_name in self.model.blocked_piles:
                possible = False

            if possible:
                possbile_actions.append(self.action_mapping_inverse[from_pile_name])

        if len(possbile_actions) == 0:
            possbile_actions.append(0)

        return possbile_actions

    def get_logs(self, path=None):
        log = self.model.monitor.get_logs(path)
        return log

    def _calculate_reward(self):
        empty_travel_time = 0.0
        avoiding_time = 0.0
        for crane_name in self.model.reward_info.keys():
            empty_travel_time += self.model.reward_info[crane_name]["Empty Travel Time"]
            avoiding_time += self.model.reward_info[crane_name]["Avoiding Time"]

        if self.model.env.now != self.time:
            reward = - (empty_travel_time + avoiding_time) / (2 * (self.model.env.now - self.time))
        else:
            reward = 0

        return reward

    def _get_state_homo(self):
        state = Data()

        num_of_node = len(self.crane_list) + len(self.pile_list)
        num_of_edge = num_of_node * (num_of_node - 1)

        node_features = np.zeros((num_of_node, 2))
        edge_index = np.zeros((2, num_of_edge))

        for i, crane_name in enumerate(self.crane_list):
            info = self.model.state_info[crane_name]
            crane_current_x = info["Current Coord"][0]
            crane_target_x = info["Target Coord"][0]

            node_features[i, 0] = crane_current_x / 44
            if not crane_target_x == -1:
                node_features[i, 1] = crane_target_x / 44

        for i, from_pile_name in enumerate(self.pile_list):
            from_pile_x = self.model.piles[from_pile_name].coord[0]
            node_features[i + len(self.crane_list), 0] = from_pile_x / 44

            plates = self.model.piles[from_pile_name].plates
            if len(plates) >= 1:
                to_pile_name = self.model.piles[from_pile_name].plates[-1].to_pile
                to_pile_x = self.model.piles[to_pile_name].coord[0]
                node_features[i + len(self.crane_list), 1] = to_pile_x / 44

        for i, crane_name in enumerate(self.crane_list):
            idx = 0
            edge_index[0, idx + i * (len(self.crane_list) - 1):idx + (i + 1) * (len(self.crane_list) - 1)] = i
            edge_index[1, idx + i * (len(self.crane_list) - 1):idx + (i + 1) * (len(self.crane_list) - 1)] \
                = [j for j in range(len(self.crane_list)) if j != i]

            idx = len(self.crane_list)
            edge_index[0, idx + i * len(self.pile_list):idx + (i + 1) * len(self.pile_list)] = i
            edge_index[1, idx + i * len(self.pile_list):idx + (i + 1) * len(self.pile_list)] \
                = range(len(self.crane_list), len(self.crane_list) + len(self.pile_list))

            idx = len(self.crane_list) + len(self.crane_list) * len(self.pile_list)
            edge_index[0, idx + i * len(self.pile_list):idx + (i + 1) * len(self.pile_list)] \
                = range(len(self.crane_list), len(self.crane_list) + len(self.pile_list))
            edge_index[1, idx + i * len(self.pile_list):idx + (i + 1) * len(self.pile_list)] = i

        idx = len(self.crane_list) + 2 * len(self.crane_list) * len(self.pile_list)
        for i, from_pile_name in enumerate(self.pile_list):
            edge_index[0, idx + i * (len(self.pile_list) - 1):idx + (i + 1) * (len(self.pile_list) - 1)] \
                = len(self.crane_list) + i
            edge_index[1, idx + i * (len(self.pile_list) - 1):idx + (i + 1) * (len(self.pile_list) - 1)] \
                = [len(self.crane_list) + j for j in range(len(self.pile_list)) if j != i]

        edge_type = torch.concat([0 * torch.ones(len(self.crane_list)),
                                  1 * torch.ones(len(self.crane_list) * len(self.pile_list)),
                                  2 * torch.ones(len(self.crane_list) * len(self.pile_list)),
                                  3 * torch.ones(len(self.pile_list) * (len(self.pile_list) - 1))])

        state.x = torch.tensor(node_features, dtype=torch.float32)
        state.edge_index = torch.tensor(edge_index, dtype=torch.long)
        state.edge_type = edge_type.type(torch.long)

        return state

    def _get_state(self):
        state = HeteroData()

        num_of_node_for_crane = len(self.crane_list)
        num_of_node_for_pile = len(self.pile_list)
        num_of_edge_for_pile_pile = len(self.pile_list) * (len(self.pile_list) - 1)
        num_of_edge_for_crane_pile = len(self.pile_list) * len(self.crane_list)
        num_of_edge_for_pile_crane = len(self.pile_list) * len(self.crane_list)
        num_of_edge_for_crane_crane = len(self.crane_list) * (len(self.crane_list) - 1)

        node_features_for_pile = np.zeros((num_of_node_for_pile, self.state_size["pile"]))
        node_features_for_crane = np.zeros((num_of_node_for_crane, self.state_size["crane"]))

        for i, crane_name in enumerate(self.crane_list):
            if crane_name in self.working_crane_ids:
                info = self.model.state_info[crane_name]
                crane_current_x = info["Current Coord"][0]
                crane_target_x = info["Target Coord"][0]
            else:
                if crane_name == "Crane-1":
                    crane_current_x = 2
                else:
                    crane_current_x = 43
                crane_target_x = -1.0

            node_features_for_crane[i, 0] = crane_current_x / 44
            if not crane_target_x == -1:
                node_features_for_crane[i, 1] = crane_target_x / 44

        for i, from_pile_name in enumerate(self.pile_list):
            from_pile_x = self.model.piles[from_pile_name].coord[0]
            plates = self.model.piles[from_pile_name].plates

            for j in range(self.look_ahead):
                node_features_for_pile[i, 3 * j] = from_pile_x / 44

                if len(plates) >= j + 1:
                    plate = self.model.piles[from_pile_name].plates[-1-j]
                    to_pile_name = plate.to_pile
                    to_pile_x = self.model.piles[to_pile_name].coord[0]
                    weight = plate.weight
                    node_features_for_pile[i, 3 * j + 1] = to_pile_x / 44
                    node_features_for_pile[i, 3 * j + 2] = weight / 19.294

        state['crane'].x = torch.tensor(node_features_for_crane, dtype=torch.float32)
        state['pile'].x = torch.tensor(node_features_for_pile, dtype=torch.float32)

        edge_pile_pile = np.zeros((2, num_of_edge_for_pile_pile))
        edge_crane_pile = np.zeros((2, num_of_edge_for_crane_pile))
        edge_pile_crane = np.zeros((2, num_of_edge_for_pile_crane))
        edge_crane_crane = np.zeros((2, num_of_edge_for_crane_crane))

        for i, from_pile_name in enumerate(self.pile_list):
            edge_pile_pile[0, i * (len(self.pile_list) - 1):(i + 1) * (len(self.pile_list) - 1)] = i
            edge_pile_pile[1, i * (len(self.pile_list) - 1):(i + 1) * (len(self.pile_list) - 1)] \
                = [j for j in range(len(self.pile_list)) if j != i]

        for i, crane_name in enumerate(self.crane_list):
            edge_crane_pile[0, i * len(self.pile_list):(i + 1) * len(self.pile_list)] = i
            edge_crane_pile[1, i * len(self.pile_list):(i + 1) * len(self.pile_list)] = range(len(self.pile_list))

            edge_pile_crane[0, i * len(self.pile_list):(i + 1) * len(self.pile_list)] = range(len(self.pile_list))
            edge_pile_crane[1, i * len(self.pile_list):(i + 1) * len(self.pile_list)] = i

            edge_crane_crane[0, i * (len(self.crane_list) - 1):(i + 1) * (len(self.crane_list) - 1)] = i
            edge_crane_crane[1, i * (len(self.crane_list) - 1):(i + 1) * (len(self.crane_list) - 1)] \
                = [j for j in range(len(self.crane_list)) if j != i]

        state['pile', 'stacking', 'pile'].edge_index = torch.tensor(edge_pile_pile, dtype=torch.long)
        state['crane', 'moving', 'pile'].edge_index = torch.tensor(edge_crane_pile, dtype=torch.long)
        state['pile', 'moving_rev', 'crane'].edge_index = torch.tensor(edge_pile_crane, dtype=torch.long)
        state['crane', 'interfering', 'crane'].edge_index = torch.tensor(edge_crane_crane, dtype=torch.long)

        return state