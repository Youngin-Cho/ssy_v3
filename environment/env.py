import torch
import random
import numpy as np

from torch_geometric.data import HeteroData
from environment.data import generate_data
from environment.simulation_v2 import Management


class SteelStockYard(object):
    def __init__(self, look_ahead=2,  # 상태 구성 시 각 파일에서 포함할 강재의 수
                 bays=("A", "B"),  # 강재적치장의 베이 이름
                 num_of_cranes=2,
                 safety_margin=5,
                 num_of_storage_to_piles=10,  # 적치 작업 시 강재를 적치할 파일의 수 (데이터 랜덤 생성 시)
                 num_of_reshuffle_from_piles=10,  # 선별 작업 시 이동할 강재가 적치된 파일의 수 (데이터 랜덤 생성 시)
                 num_of_reshuffle_to_piles=20,  # 선별 작업 시 강재가 이동할 파일의 수 (데이터 랜덤 생성 시)
                 num_of_retrieval_from_piles=4,  # 출고 작업 시 이동할 강재가 적치된 파일의 수 (데이터 랜덤 생성 시)
                 df_storage=None,
                 df_reshuffle=None,
                 df_retrieval=None):

        self.look_ahead = look_ahead
        self.bays = bays
        self.num_of_cranes = num_of_cranes
        self.safety_margin = safety_margin
        self.num_of_storage_to_piles = num_of_storage_to_piles
        self.num_of_reshuffle_from_piles = num_of_reshuffle_from_piles
        self.num_of_reshuffle_to_piles = num_of_reshuffle_to_piles
        self.num_of_retrieval_from_piles = num_of_retrieval_from_piles

        self.action_size = 2 * 40 + 2 + 1
        self.state_size = {"crane": 2 * len(bays) * 44, "pile": len(bays) * 44, "plate": len(bays) * 44}
        self.meta_data = (["crane", "pile", "plate"],
                          [("crane", "interfering", "crane"),
                           ("plate", "stacking", "plate"),
                           ("plate", "locating", "pile"),
                           ("pile", "moving", "crane")])

        if (df_storage is None) and (df_reshuffle is None) and (df_retrieval is None):
            self.random_data = True
            self.df_storage, self.df_reshuffle, self.df_retrieval = generate_data(self.num_of_storage_to_piles,
                                                                                  self.num_of_reshuffle_from_piles,
                                                                                  self.num_of_reshuffle_to_piles,
                                                                                  self.num_of_retrieval_from_piles,
                                                                                  self.bays)
        else:
            self.random_data = False
            self.df_storage, self.df_reshuffle, self.df_retrieval = df_storage, df_reshuffle, df_retrieval

        self.pile_list = list(self.df_storage["pileno"].unique()) + list(self.df_reshuffle["pileno"].unique())
        self.model = Management(self.df_storage, self.df_reshuffle, self.df_retrieval,
                                bays=self.bays, safety_margin=self.safety_margin)

        self.action_mapping = {i + 1: pile_name for i, pile_name in enumerate(self.model.piles.keys())}
        self.action_mapping_inverse = {y: x for x, y in self.action_mapping.items()}
        self.crane_in_decision = None
        self.safety_margin = safety_margin

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

        return next_state, reward, done, info

    def reset(self):
        if self.random_data:
            self.df_storage, self.df_reshuffle, self.df_retrieval = generate_data(self.num_of_storage_to_piles,
                                                                                  self.num_of_reshuffle_from_piles,
                                                                                  self.num_of_reshuffle_to_piles,
                                                                                  self.num_of_retrieval_from_piles,
                                                                                  self.bays)

        self.model = Management(self.df_storage, self.df_reshuffle, self.df_retrieval,
                                bays=self.bays, safety_margin=self.safety_margin)
        self.pile_list = list(self.df_storage["pileno"].unique()) + list(self.df_reshuffle["pileno"].unique())
        self.crane_list = [crane.name for crane in self.model.cranes.items]

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

            if self.crane_in_decision == 0 and from_pile_x > 43 - self.safety_margin:
                possible = False
            if self.crane_in_decision == 1 and from_pile_x < 1 + self.safety_margin:
                possible = False
            if self.crane_in_decision == 0 and to_pile_x > 43 - self.safety_margin:
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
        # wasting_time = 0
        # for crane_name in self.model.reward_info.keys():
        #     wasting_time += self.model.reward_info[crane_name]["Wasting Time"]
        # reward = 1 / wasting_time if wasting_time != 0 else 1
        if self.crane_in_decision == 0:
            increased_time = self.model.env.now - self.model.crane1_decision_time
            reward = - self.model.reward_info["Crane-1"]["Wasting Time"]
        else:
            increased_time = self.model.env.now - self.model.crane2_decision_time
            reward = - self.model.reward_info["Crane-2"]["Wasting Time"]
        # reward = 1 / increased_time if increased_time != 0 else 0
        # reward = 1 / reward if reward != 0 else 1

        return reward / 87

    def _get_state(self):
        state = HeteroData()

        num_of_node_for_crane = self.num_of_cranes
        num_of_node_for_pile = len(self.pile_list)
        num_of_node_for_plate = self.look_ahead * len(self.pile_list)
        num_of_edge_for_plate_plate = (self.look_ahead - 1) * len(self.pile_list)
        num_of_edge_for_plate_pile = len(self.pile_list)
        num_of_edge_for_pile_crane = len(self.pile_list) * self.num_of_cranes
        num_of_edge_for_crane_crane = np.cumsum(range(self.num_of_cranes))[-1] * 2

        node_features_for_plate = np.zeros((num_of_node_for_plate, self.state_size["plate"]))
        node_features_for_pile = np.zeros((num_of_node_for_pile, self.state_size["pile"]))
        node_features_for_crane = np.zeros((num_of_node_for_crane, self.state_size["crane"]))

        all_x_coords = np.array([i for i in range(1, 45)] * len(self.bays))
        all_y_coords = np.array([1 for _ in range(1, 45)] + [2 for _ in range(1, 45)])

        for i, crane_name in enumerate(self.crane_list):
            info = self.model.state_info[crane_name]
            crane_current_x = info["Current Coord"][0]
            crane_current_y = info["Current Coord"][1]
            crane_target_x = info["Target Coord"][0]
            crane_target_y = info["Target Coord"][1]
            features = np.maximum(2 * np.abs(all_x_coords - crane_current_x), np.abs(all_y_coords - crane_current_y))
            node_features_for_crane[i, :len(all_x_coords)] = features / 86
            if crane_target_x == -1 and crane_target_y == -1:
                features = np.maximum(2 * np.abs(all_x_coords - crane_target_x), np.abs(all_y_coords - crane_target_y))
                node_features_for_crane[i, len(all_x_coords):] = features / 86

        for i, from_pile_name in enumerate(self.pile_list):
            from_pile_x = self.model.piles[from_pile_name].coord[0]
            from_pile_y = self.model.piles[from_pile_name].coord[1]
            features = np.maximum(2 * np.abs(all_x_coords - from_pile_x), np.abs(all_y_coords - from_pile_y))
            node_features_for_pile[i] = features / 86

            plates = self.model.piles[from_pile_name].plates
            for j in range(self.look_ahead):
                if len(plates) >= j + 1:
                    to_pile_name = self.model.piles[from_pile_name].plates[-1 - j].to_pile
                    to_pile_x = self.model.piles[to_pile_name].coord[0]
                    to_pile_y = self.model.piles[to_pile_name].coord[1]
                    features = np.maximum(2 * np.abs(all_x_coords - to_pile_x), np.abs(all_y_coords - to_pile_y))
                    node_features_for_plate[self.look_ahead * i + j] = features / 86

        state['crane'].x = torch.tensor(node_features_for_crane, dtype=torch.float32)
        state['pile'].x = torch.tensor(node_features_for_pile, dtype=torch.float32)
        state['plate'].x = torch.tensor(node_features_for_plate, dtype=torch.float32)

        edge_plate_plate = np.zeros((2, num_of_edge_for_plate_plate))
        edge_plate_pile = np.zeros((2, num_of_edge_for_plate_pile))
        edge_pile_crane = np.zeros((2, num_of_edge_for_pile_crane))
        edge_crane_crane = np.zeros((2, num_of_edge_for_crane_crane))

        for i in range(self.look_ahead - 1):
            edge_plate_plate[0, i * len(self.pile_list):(i + 1) * len(self.pile_list)] \
                = range(i + 1, self.look_ahead * len(self.pile_list), self.look_ahead)
            edge_plate_plate[1, i * len(self.pile_list):(i + 1) * len(self.pile_list)] \
                = range(i, self.look_ahead * len(self.pile_list), self.look_ahead)

        edge_plate_pile[0, :] = range(0, self.look_ahead * len(self.pile_list), self.look_ahead)
        edge_plate_pile[1, :] = range(len(self.pile_list))

        for i, crane_name in enumerate(self.crane_list):
            edge_pile_crane[0, i * len(self.pile_list):(i + 1) * len(self.pile_list)] = range(len(self.pile_list))
            edge_pile_crane[1, i * len(self.pile_list):(i + 1) * len(self.pile_list)] = i

        for i, crane_name in enumerate(self.crane_list):
            edge_crane_crane[0, i * len(self.crane_list):(i + 1) * (len(self.crane_list) - 1)] = i
            edge_crane_crane[1, i * len(self.crane_list):(i + 1) * (len(self.crane_list) - 1)] \
                = [j for j in range(len(self.crane_list)) if j != i]

        state['plate', 'stacking', 'plate'].edge_index = torch.tensor(edge_plate_plate, dtype=torch.long)
        state['plate', 'locating', 'pile'].edge_index = torch.tensor(edge_plate_pile, dtype=torch.long)
        state['pile', 'moving', 'crane'].edge_index = torch.tensor(edge_pile_crane, dtype=torch.long)
        state['crane', 'interfering', 'crane'].edge_index = torch.tensor(edge_crane_crane, dtype=torch.long)

        return state