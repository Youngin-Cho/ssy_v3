import torch
import numpy as np
import pandas as pd

from torch_geometric.data import HeteroData
from environment.data import DataGenerator
from environment.simulation import Management


class SteelStockYard:
    def __init__(self, data_src, look_ahead=2, max_x=44, max_y=2, row_range=("A", "B"), bay_range=(1, 40),
                 input_points=(1,), output_points=(23, 27, 44), working_crane_ids=("Crane-1", "Crane-2"),
                 safety_margin=5, multi_num=3, multi_w=20.0, multi_dis=2,
                 reward_sig=0, rl=-True, record_events=False, device=None):

        self.data_src = data_src
        self.look_ahead = look_ahead
        self.max_x = max_x
        self.max_y = max_y
        self.row_range = row_range
        self.bay_range = bay_range
        self.input_points = input_points
        self.output_points = output_points
        self.working_crane_ids = working_crane_ids
        self.safety_margin = safety_margin
        self.multi_num = multi_num
        self.multi_w = multi_w
        self.multi_dis = multi_dis
        self.reward_sig = reward_sig
        self.rl = rl
        self.record_events = record_events
        self.device = device

        self.x_velocity = 0.5
        self.y_velocity = 1.0

        if type(self.data_src) is DataGenerator:
            self.df_storage, self.df_reshuffle, self.df_retrieval = self.data_src.generate()
        else:
            self.df_storage = pd.read_excel(self.data_src, sheet_name="storage", engine="openpyxl")
            self.df_reshuffle = pd.read_excel(self.data_src, sheet_name="reshuffle", engine="openpyxl")
            self.df_retrieval = pd.read_excel(self.data_src, sheet_name="retrieval", engine="openpyxl")

        self.crane_list = ["Crane-1", "Crane-2"]
        self.pile_list = list(self.df_storage["pileno"].unique()) + list(self.df_reshuffle["pileno"].unique())
        self.model = Management(self.df_storage, self.df_reshuffle, self.df_retrieval,
                                max_x=self.max_x, max_y=self.max_y, row_range=self.row_range, bay_range=self.bay_range,
                                input_points=self.input_points, output_points=self.output_points,
                                working_crane_ids=self.working_crane_ids, safety_margin=self.safety_margin,
                                multi_num=self.multi_num, multi_w=self.multi_w, multi_dis=self.multi_dis,
                                record_events=self.record_events)

        self.action_size = len(self.pile_list) * len(self.crane_list)
        self.state_size = {"crane": 2, "pile": 1 + 1 * look_ahead}
        self.meta_data = (["crane", "pile"],
                          [("pile", "moving_rev", "crane"),
                           ("crane", "moving", "pile")])
        self.num_nodes = {"crane": len(self.crane_list), "pile": len(self.pile_list)}

        self.action_mapping = {i: pile_name for i, pile_name in enumerate(self.pile_list)}
        self.action_mapping_inverse = {y: x for x, y in self.action_mapping.items()}
        self.crane_in_decision = None
        self.time = 0.0

    def step(self, action):
        crane_id = action % self.num_nodes["crane"]
        pile_id = action // self.num_nodes["crane"]
        pile_name = self.action_mapping[pile_id]
        done = False

        self.model.do_action.succeed(pile_name)
        self.model.decision_time = False

        while True:
            while True:
                if self.model.decision_time:
                    break
                else:
                    if len(self.model.move_list) == 0:
                        done = True
                        break
                    else:
                        if len(self.model.env._queue) == 0:
                            log = self.get_logs()
                            print(0)
                        self.model.env.step()

            if self.rl:
                next_state, mask = self._get_state()
            else:
                next_state, mask = self._get_state_for_heuristics()
            self.crane_in_decision = self.model.crane_in_decision.id

            if not mask.any():
                self.model.do_action.succeed("Waiting")
                self.model.decision_time = False
            else:
                break

        reward = self._calculate_reward()
        self.time = self.model.env.now

        return next_state, reward, done, mask

    def reset(self):
        self.model = Management(self.df_storage, self.df_reshuffle, self.df_retrieval,
                                max_x=self.max_x, max_y=self.max_y, row_range=self.row_range, bay_range=self.bay_range,
                                input_points=self.input_points, output_points=self.output_points,
                                working_crane_ids=self.working_crane_ids, safety_margin=self.safety_margin,
                                multi_num=self.multi_num, multi_w=self.multi_w, multi_dis=self.multi_dis,
                                record_events=self.record_events)
        while True:
            while True:
                if self.model.decision_time:
                    break
                else:
                    self.model.env.step()

            if self.rl:
                initial_state, mask = self._get_state()
            else:
                initial_state, mask = self._get_state_for_heuristics()
            self.crane_in_decision = self.model.crane_in_decision.id

            if not mask.any():
                self.model.do_action.succeed("Waiting")
                self.model.decision_time = False
            else:
                break

        return initial_state, mask

    def get_logs(self, path=None):
        log = self.model.monitor.get_logs(path)
        return log

    def _calculate_reward(self):
        empty_travel_time = 0.0
        avoiding_time = 0.0
        waiting_time = 0.0
        for crane_name in self.model.reward_info.keys():
            empty_travel_time += self.model.reward_info[crane_name]["Empty Travel Time"]
            avoiding_time += self.model.reward_info[crane_name]["Avoiding Time"]
            waiting_time += self.model.reward_info[crane_name]["Waiting Time"]

        if self.model.env.now != self.time:
            if self.reward_sig == 0:
                reward = - (empty_travel_time + avoiding_time + waiting_time) / (2 * (self.model.env.now - self.time))
            elif self.reward_sig == 1:
                reward = - empty_travel_time / (2 * (self.model.env.now - self.time))
            elif self.reward_sig == 2:
                reward = - avoiding_time / (2 * (self.model.env.now - self.time))
            else:
                reward = reward = - waiting_time / (2 * (self.model.env.now - self.time))
        else:
            reward = 0

        return reward

    def _get_state(self):
        state = HeteroData()

        X_piles = np.zeros((self.num_nodes["pile"], self.state_size["pile"]))
        X_cranes = np.zeros((self.num_nodes["crane"], self.state_size["crane"]))

        edge_pile_to_crane, edge_crane_to_pile = [[], []], [[], []]

        mask_piles = np.zeros((self.num_nodes["crane"], self.num_nodes["pile"]), dtype=bool)
        mask_cranes = np.zeros((self.num_nodes["crane"], self.num_nodes["pile"]), dtype=bool)
        mask_eligibility = np.zeros((self.num_nodes["crane"], self.num_nodes["pile"]), dtype=bool)

        for i, crane_name in enumerate(self.crane_list):
            if crane_name in self.working_crane_ids:
                info = self.model.state_info[crane_name]
                crane_current_x = info["Current Coord"][0]
                crane_target_x = info["Target Coord"][0]
                if info["Idle"] and self.model.crane_in_decision.name == crane_name:
                    mask_cranes[i, :] = 1
            else:
                if crane_name == "Crane-1":
                    crane_current_x = 2
                else:
                    crane_current_x = 43
                crane_target_x = -1.0

            X_cranes[i, 0] = crane_current_x / 44
            if not crane_target_x == -1:
                X_cranes[i, 1] = crane_target_x / 44

        for i, from_pile_name in enumerate(self.pile_list):
            from_pile_x = self.model.piles[from_pile_name].coord[0]
            plates = self.model.piles[from_pile_name].plates

            cnt1 = self.model.state_info["Crane-1"]["Locations"].count(from_pile_name)
            cnt2 = self.model.state_info["Crane-2"]["Locations"].count(from_pile_name)

            if (cnt1 == 0) and (cnt2 == 0) and (not from_pile_name in self.model.blocked_piles):
                mask_piles[:, i] = 1

            X_piles[i, 0] = from_pile_x / 44
            for j in range(self.look_ahead):
                if len(plates) >= j + 1:
                    plate = self.model.piles[from_pile_name].plates[-1-j]
                    to_pile_name = plate.to_pile
                    to_pile_x = self.model.piles[to_pile_name].coord[0]
                    # weight = plate.w
                    X_piles[i, 1 * j + 1] = to_pile_x / 44
                    # node_features_for_pile[i, 2 * j + 2] = weight / 19.294

                    if j == 0:
                        if (from_pile_x <= self.max_x - self.safety_margin) and (to_pile_x <= self.max_x - self.safety_margin):
                            mask_eligibility[0, i] = 1
                        if (from_pile_x >= 1 + self.safety_margin) and ( to_pile_x >= 1 + self.safety_margin):
                            mask_eligibility[1, i] = 1

        state['crane'].x = torch.tensor(X_cranes, dtype=torch.float32).to(self.device)
        state['pile'].x = torch.tensor(X_piles, dtype=torch.float32).to(self.device)

        mask = mask_piles & mask_cranes & mask_eligibility

        for i, crane_name in enumerate(self.crane_list):
            for j, from_pile_name in enumerate(self.pile_list):
                from_pile_x = self.model.piles[from_pile_name].coord[0]
                if len(self.model.piles[from_pile_name].plates) > 0:
                    to_pile_name = self.model.piles[from_pile_name].plates[-1].to_pile
                    to_pile_x = self.model.piles[to_pile_name].coord[0]
                else:
                    to_pile_x = 0 if i == 0 else self.max_x

                possible = True
                if i == 0 and from_pile_x > self.max_x - self.safety_margin:
                    possible = False
                if i == 1 and from_pile_x < 1 + self.safety_margin:
                    possible = False
                if i == 0 and to_pile_x > self.max_x - self.safety_margin:
                    possible = False
                if i == 1 and to_pile_x < 1 + self.safety_margin:
                    possible = False
                if possible:
                    edge_crane_to_pile[0].append(i)
                    edge_crane_to_pile[1].append(j)
                    edge_pile_to_crane[0].append(j)
                    edge_pile_to_crane[1].append(i)

        state['crane', 'moving', 'pile'].edge_index = torch.tensor(np.array(edge_crane_to_pile), dtype=torch.long).to(self.device)
        state['pile', 'moving_rev', 'crane'].edge_index = torch.tensor(np.array(edge_pile_to_crane), dtype=torch.long).to(self.device)

        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        return state, mask

    def _get_state_for_heuristics(self):
        state = {"action_mapping": self.action_mapping,
                 "piles_all": [], "piles_sd": [], "piles_ma": [], "piles_mx": []}

        mask = np.zeros((self.num_nodes["crane"], self.num_nodes["pile"]), dtype=bool)

        moving_time_min = float('inf')
        moving_time_min_wo_interference = float('inf')
        for i, from_pile_name in enumerate(self.pile_list):
            from_pile_x = self.model.piles[from_pile_name].coord[0]
            from_pile_y = self.model.piles[from_pile_name].coord[1]
            plates = self.model.piles[from_pile_name].plates

            if len(plates) > 0:
                plate = plates[-1]
                to_pile_name = plate.to_pile
                to_pile_x = self.model.piles[to_pile_name].coord[0]
                to_pile_y = self.model.piles[to_pile_name].coord[1]
            else:
                continue

            for j, crane_name in enumerate(self.crane_list):
                info = self.model.state_info[crane_name]
                if crane_name == "Crane-1":
                    opposite_info = self.model.state_info["Crane-2"]
                else:
                    opposite_info = self.model.state_info["Crane-1"]
                crane_current_x = info["Current Coord"][0]
                crane_current_y = info["Current Coord"][1]

                moving_time_crane = max(abs(from_pile_x - crane_current_x) / self.x_velocity,
                                        abs(from_pile_y - crane_current_y) / self.y_velocity)

                cnt = opposite_info["Locations"].count(from_pile_name)

                if (cnt == 0) and (not from_pile_name in self.model.blocked_piles):
                    if info["Idle"] and self.model.crane_in_decision.name == crane_name:
                        expression1 = ((crane_name == "Crane-1")
                                       and ((from_pile_x <= self.max_x - self.safety_margin)
                                       and (to_pile_x <= self.max_x - self.safety_margin)))
                        expression2 = ((crane_name == "Crane-2")
                                       and ((from_pile_x >= 1 + self.safety_margin)
                                       and (to_pile_x >= 1 + self.safety_margin)))
                        if expression1 or expression2:
                            mask[j, i] = 1
                            state["piles_all"].append((i, j))
                            if moving_time_crane <= moving_time_min:
                                if moving_time_crane < moving_time_min:
                                    state["piles_sd"] = []
                                    moving_time_min = moving_time_crane
                                state["piles_sd"].append((i, j))

                            if opposite_info["Idle"]:
                                state["piles_ma"].append((i, j))
                                if moving_time_crane <= moving_time_min_wo_interference:
                                    if moving_time_crane < moving_time_min_wo_interference:
                                        state["piles_mx"] = []
                                        moving_time_min_wo_interference = moving_time_crane
                                    state["piles_mx"].append((i, j))
                            else:
                                opposite_crane_current_x = opposite_info["Current Coord"][0]
                                opposite_crane_current_y = opposite_info["Current Coord"][1]
                                opposite_crane_target_x = opposite_info["Target Coord"][0]
                                opposite_crane_target_y = opposite_info["Target Coord"][1]

                                moving_time_opposite_crane = max(abs(opposite_crane_target_x - opposite_crane_current_x) / self.x_velocity,
                                                                 abs(opposite_crane_target_y - opposite_crane_current_y) / self.y_velocity)

                                moving_time = min(moving_time_crane, moving_time_opposite_crane)

                                xcoord_crane = crane_current_x + moving_time * self.x_velocity * np.sign(from_pile_x - crane_current_x)
                                xcoord_opposite_crane = opposite_crane_current_x + moving_time * self.x_velocity * np.sign(opposite_crane_target_x - opposite_crane_current_x)

                                if ((crane_name == "Crane-1" and xcoord_crane > xcoord_opposite_crane - self.safety_margin)
                                        or (crane_name == "Crane-2" and xcoord_crane < xcoord_opposite_crane + self.safety_margin)):
                                    pass
                                else:
                                    state["piles_ma"].append((i, j))
                                    if moving_time_crane <= moving_time_min_wo_interference:
                                        if moving_time_crane < moving_time_min_wo_interference:
                                            state["piles_mx"] = []
                                            moving_time_min_wo_interference = moving_time_crane
                                        state["piles_mx"].append((i, j))

        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        return state, mask