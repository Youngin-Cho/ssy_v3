import simpy
import random
import numpy as np
import pandas as pd

from collections import OrderedDict


class PriorityFilterGet(simpy.resources.store.FilterStore.get):
    def __init__(self, resource, filt, priority=10, preempt=True):
        self.priority = priority
        self.preempt = preempt
        self.time = resource._env.now
        self.usage_since = None
        self.key = (self.priority, self.time, not self.preempt)
        super().__init__(resource, filt)


class PriorityFilterStore(simpy.resources.store.FilterStore):
    GetQueue = simpy.resources.resource.SortedQueue
    get = simpy.core.BoundClass(PriorityFilterGet)


# class PriorityGet(simpy.resources.base.Get):
#     def __init__(self, resource, filter, priority=10, preempt=True):
#         self.priority = priority
#         self.preempt = preempt
#         self.time = resource._env.now
#         self.usage_since = None
#         self.key = (self.priority, self.time, not self.preempt)
#         super().__init__(resource)
#
#
# class PriorityBaseStore(simpy.resources.store.Store):
#
#     GetQueue = simpy.resources.resource.SortedQueue
#     get = simpy.core.BoundClass(PriorityGet)


class Plate:
    def __init__(self, name, from_pile, to_pile, w):
        self.name = name
        self.from_pile = from_pile
        self.to_pile = to_pile
        self.w = w


class Pile:
    def __init__(self, name, coord):
        self.name = name
        self.coord = coord
        self.plates = list()
        self.blocking = list()

        self.end_plates = list()


class Crane:
    def __init__(self, env, name, id, safety_margin, w_limit, coord):
        self.env = env
        self.name = name
        self.id = id
        self.safety_margin = safety_margin
        self.w_limit = w_limit
        self.current_coord = coord
        self.target_coord = (-1.0, -1.0)
        self.safety_xcoord = -1.0

        self.from_piles = list()
        self.to_piles = list()
        self.plates = list()

        self.opposite = None
        self.idle = True
        self.move_until = -1.0
        self.idle_event = None
        # self.avoiding_event = None

        self.moving = False
        self.loading = False
        self.unloading = False
        self.x_velocity = 0.5
        self.y_velocity = 1.0
        self.update_time = 0.0
        self.waiting_time = 0.0
        self.empty_travel_time = 0.0
        self.avoiding_time = 0.0

    def get_plate(self, pile):
        plate = pile.plates.pop()
        self.plates.append(plate)
        return plate.name

    def put_plate(self, pile):
        plate = self.plates.pop()
        pile.end_plates.append(plate)
        return plate.name

    def move(self, to_xcoord=None, to_ycoord=None):
        self.moving = True

        moving_time = self.get_moving_time(to_xcoord, to_ycoord)
        if self.opposite.idle:
            xcoord_crane = self.current_coord[0] + moving_time * self.x_velocity \
                           * np.sign(self.target_coord[0] - self.current_coord[0])
            xcoord_opposite_crane = self.opposite.current_coord[0]
            if self.name == 'Crane-1' and xcoord_crane > xcoord_opposite_crane - self.safety_margin:
                moving_time_opposite = self.opposite.get_moving_time(to_xcoord=xcoord_crane + self.safety_margin)
                self.opposite.move_until = self.env.now + moving_time_opposite
            elif self.name == 'Crane-2' and xcoord_crane < xcoord_opposite_crane + self.safety_margin:
                moving_time_opposite = self.opposite.get_moving_time(to_xcoord=xcoord_crane - self.safety_margin)
                self.opposite.move_until = self.env.now + moving_time_opposite

        yield self.env.timeout(moving_time)
        self.opposite.update_location(self.env.now)
        self.update_location(self.env.now, on_pile=True)

        self.moving = False

        return moving_time

    def get_moving_time(self, to_xcoord=None, to_ycoord=None):
        x_moving_time = abs(to_xcoord - self.current_coord[0]) / self.x_velocity
        if to_ycoord is not None:
            y_moving_time = abs(to_ycoord - self.current_coord[1]) / self.y_velocity
        else:
            if self.idle:
                y_moving_time = 0.0
            else:
                y_moving_time = abs(self.target_coord[1] - self.current_coord[1]) / self.y_velocity
        moving_time = max(x_moving_time, y_moving_time)
        return int(moving_time)

    def update_location(self, time, on_pile=False):
        x_coord = self.current_coord[0]
        y_coord = self.current_coord[1]
        time_elapsed = time - self.update_time

        if time_elapsed > 0.0:
            if self.idle:
                current_xcoord_crane = self.current_coord[0]
                target_xcoord_opposite_crane = self.opposite.target_coord[0]

                if self.name == 'Crane-1':
                    if current_xcoord_crane >= target_xcoord_opposite_crane:
                        x_coord = target_xcoord_opposite_crane - self.safety_margin
                else:
                    if current_xcoord_crane <= target_xcoord_opposite_crane:
                        x_coord = target_xcoord_opposite_crane + self.safety_margin
            else:
                if self.moving:
                    if self.safety_xcoord != -1.0:
                        x_coord = x_coord + time_elapsed * self.x_velocity * np.sign(self.safety_xcoord - x_coord)
                    else:
                        x_coord = x_coord + time_elapsed * self.x_velocity * np.sign(self.target_coord[0] - x_coord)
                    y_coord = y_coord + time_elapsed * self.y_velocity * np.sign(self.target_coord[1] - y_coord)
                    x_coord = np.clip(x_coord, 1, 44)
                    y_coord = np.clip(y_coord, 1, 2)
                else:
                    y_coord = y_coord + time_elapsed * self.y_velocity * np.sign(self.target_coord[1] - y_coord)
                    y_coord = np.clip(y_coord, 1, 2)

        self.update_time = time
        self.current_coord = (x_coord, y_coord)

        if on_pile:
            self.current_coord = (int(x_coord), int(y_coord))


class Conveyor:
    def __init__(self, name, coord, IAT):
        self.name = name
        self.coord = [coord]
        self.IAT = IAT

        self.end_plates = list()


class Monitor:
    def __init__(self, record_events=False):
        self.record_events = record_events

        self.time = []
        self.event = []
        self.crane = []
        self.location = []
        self.plate = []
        self.tag = []

    def record(self, time, event, crane=None, location=None, plate=None, tag=None):
        self.time.append(time)
        self.event.append(event)
        self.crane.append(crane)
        self.location.append(location)
        self.plate.append(plate)
        self.tag.append(tag)

    def get_logs(self, file_path=None):
        records = pd.DataFrame(columns=["Time", "Event", "Crane", "Location", "Plate", "Tag"])
        records["Time"] = self.time
        records["Event"] = self.event
        records["Crane"] = self.crane
        records["Location"] = self.location
        records["Plate"] = self.plate
        records["Tag"] = self.tag

        if file_path is not None:
            records.to_csv(file_path, index=False)

        return records


class Management:
    def __init__(self, df_storage, df_reshuffle, df_retrieval,
                 max_x=44, max_y=2, row_range=("A", "B"), bay_range=(1, 40),
                 input_points=(1,), output_points=(23, 27, 44),
                 working_crane_ids=("Crane-1", "Crane-2"), safety_margin=5,
                 multi_num=3, multi_w=20.0, multi_dis=2, record_events=False):
        self.df_storage = df_storage
        self.df_reshuffle = df_reshuffle
        self.df_retrieval = df_retrieval
        self.max_x = max_x
        self.max_y = max_y
        self.working_crane_ids = working_crane_ids
        self.row_range = row_range
        self.bay_range = bay_range
        self.input_points = input_points
        self.output_points = output_points
        self.safety_margin = safety_margin
        self.record_events = record_events

        self.multi_num = multi_num
        self.multi_w = multi_w
        self.multi_dist = multi_dis

        self.env, self.piles, self.conveyors, self.cranes, self.monitor = self._modeling()
        self.move_list = list(df_storage["pileno"].values) + list(df_reshuffle["pileno"].values)
        self.num_plates_cum = 0
        self.blocked_piles = list()
        self.last_action = None
        self.waiting_crane = None
        self.state_info = {crane.name: {"Current Coord": crane.current_coord,
                                        "Target Coord": (-1.0, -1.0)} for crane in self.cranes.items}
        self.reward_info = {crane.name: {"Empty Travel Time": 0.0,
                                         "Avoiding Time": 0.0,
                                         "Waiting Time": 0.0,
                                         "Empty Travel Time Cumulative": 0.0,
                                         "Avoiding Time Cumulative": 0.0,
                                         "Waiting Time Cumulative": 0.0} for crane in self.cranes.items}

        self.location_mapping = {tuple(pile.coord): pile for pile in self.piles.values()}  # coord를 통해 pile 호출
        for conveyor in self.conveyors.values():
            self.location_mapping[tuple(conveyor.coord + [1])] = conveyor
            self.location_mapping[tuple(conveyor.coord + [2])] = conveyor

        self.decision_time = False
        self.crane_in_decision = None
        self.crane1_decision_time = 0.0
        self.crane2_decision_time = 0.0
        self.priority_queue = []
        self.do_action = self.env.event()

        self.action = self.env.process(self.run())
        if df_retrieval is not None:
            self.action_conveyor = [self.env.process(self.release(cn)) for cn in self.conveyors.values()]

    def _modeling(self):
        env = simpy.Environment()

        row_list = [chr(i) for i in range(ord(self.row_range[0]), ord(self.row_range[1]) + 1)]
        bay_list = [i for i in range(self.bay_range[0] - 1, self.bay_range[1] + 1)]
        piles = OrderedDict()
        for i, row_id in enumerate(row_list):
            for j, bay_id in enumerate(bay_list):
                name = row_id + str(bay_id).rjust(2, '0')
                x_coord = j + 1
                y_coord = i + 1
                if self.output_points[0] <= x_coord:
                    x_coord += 1
                if self.output_points[1] <= x_coord:
                    x_coord += 1
                pile = Pile(name, (x_coord, y_coord))
                piles[name] = pile

        conveyors = OrderedDict()
        conveyors['cn1'] = Conveyor('cn1', self.output_points[0], 0.01)
        conveyors['cn2'] = Conveyor('cn2', self.output_points[1], 0.01)
        conveyors['cn3'] = Conveyor('cn3', self.output_points[2], 0.0005)

        # 적치 작업 대상 강재 데이터
        self.df_storage = self.df_storage.sort_values(by=["pileno", "pileseq"])
        self.df_storage = self.df_storage.reset_index(drop=True)
        for i, row in self.df_storage.iterrows():
            piles[row["pileno"]].plates.append(Plate(row["markno"], row["pileno"], row["topile"], row["unitw"]))

        # 선별 작업 대상 강재 데이터
        self.df_reshuffle = self.df_reshuffle.sort_values(by=["pileno", "pileseq"])
        self.df_reshuffle = self.df_reshuffle.reset_index(drop=True)
        for i, row in self.df_reshuffle.iterrows():
            piles[row["pileno"]].plates.append(Plate(row["markno"], row["pileno"], row["topile"], row["unitw"]))

        # 출고 작업 대상 강재 데이터
        self.df_retrieval = self.df_retrieval.sort_values(by=["pileno", "pileseq"])
        self.df_retrieval = self.df_retrieval.reset_index(drop=True)
        for i, row in self.df_retrieval.iterrows():
            piles[row["pileno"]].plates.append(Plate(row["markno"], row["pileno"], row["topile"], row["unitw"]))

        cranes = PriorityFilterStore(env)

        crane1 = Crane(env, 'Crane-1', 0, self.safety_margin, 8.0, (2, 1))
        crane2 = Crane(env, 'Crane-2', 1, self.safety_margin, 8.0, (self.max_x - 1, 1))

        crane1.opposite = crane2
        crane2.opposite = crane1

        if crane1.name in self.working_crane_ids:
            cranes.put(crane1)
        if crane2.name in self.working_crane_ids:
            cranes.put(crane2)

        monitor = Monitor(record_events=self.record_events)

        return env, piles, conveyors, cranes, monitor

    def run(self):
        while self.move_list:
            self.check_blocked_piles()

            # request a crane
            crane = yield self.cranes.get(priority=3, filt=lambda x: True)
            self.crane_in_decision = crane
            if self.crane_in_decision.id == 0:
                self.crane1_decision_time = self.env.now
            else:
                self.crane2_decision_time = self.env.now

            self.state_info[crane.name]["Current Coord"] = crane.current_coord
            self.state_info[crane.name]["Target Coord"] = crane.target_coord
            if crane.opposite.name in self.working_crane_ids:
                self.state_info[crane.opposite.name]["Current Coord"] = crane.opposite.current_coord
                self.state_info[crane.opposite.name]["Target Coord"] = crane.opposite.target_coord

            # 크레인 대기 시간 기록
            self.reward_info[crane.name]["Waiting Time"] \
                = crane.waiting_time - self.reward_info[crane.name]["Waiting Time Cumulative"]
            self.reward_info[crane.name]["Waiting Time Cumulative"] = crane.waiting_time
            # 크레인의 empty travel time 기록
            self.reward_info[crane.name]["Empty Travel Time"] \
                = crane.empty_travel_time - self.reward_info[crane.name]["Empty Travel Time Cumulative"]
            self.reward_info[crane.name]["Empty Travel Time Cumulative"] = crane.empty_travel_time
            # 크레인의 avoiding time 기록
            self.reward_info[crane.name]["Avoiding Time"] \
                = crane.avoiding_time - self.reward_info[crane.name]["Avoiding Time Cumulative"]
            self.reward_info[crane.name]["Avoiding Time Cumulative"] = crane.avoiding_time

            if crane.opposite.name in self.working_crane_ids:
                self.reward_info[crane.opposite.name]["Waiting Time"] \
                    = crane.opposite.waiting_time - self.reward_info[crane.opposite.name]["Waiting Time Cumulative"]
                self.reward_info[crane.opposite.name]["Waiting Time Cumulative"] = crane.opposite.waiting_time
                self.reward_info[crane.opposite.name]["Empty Travel Time"] \
                    = crane.opposite.empty_travel_time - self.reward_info[crane.opposite.name]["Empty Travel Time Cumulative"]
                self.reward_info[crane.opposite.name]["Empty Travel Time Cumulative"] = crane.opposite.empty_travel_time
                self.reward_info[crane.opposite.name]["Avoiding Time"] \
                    = crane.opposite.avoiding_time - self.reward_info[crane.opposite.name]["Avoiding Time Cumulative"]
                self.reward_info[crane.opposite.name]["Avoiding Time Cumulative"] = crane.opposite.avoiding_time

            # 행동 선택을 위한 이벤트 생성
            self.decision_time = True
            self.do_action = self.env.event()
            action = yield self.do_action
            self.last_action = action
            self.do_action = None

            if action != "Waiting":
                self.multi_loading(action, crane)
            self.env.process(self.crane_run(crane))

        for action in self.action_conveyor:
            action.interrupt()

    def multi_loading(self, action, crane):
        possible_dict = dict()
        possible_dict[action] = 1

        for i in range(1, self.multi_dist + 1):
            for dx in [-i, i]:
                from_coord = (int(self.piles[action].coord[0] + dx), int(self.piles[action].coord[1]))
                if from_coord in self.location_mapping.keys():
                    if not self.location_mapping[from_coord].name in ["cn1", "cn2", "cn3"]:
                        if ((crane.name == "Crane-1" and from_coord[0] <= self.max_x - self.safety_margin) or
                                (crane.name == "Crane-2" and from_coord[0] >= 1 + self.safety_margin)):
                            possible_dict[self.location_mapping[from_coord].name] = 1

        from_list = [action]
        weight = self.piles[action].plates[-possible_dict[action]].w
        current_to_pile = self.piles[action].plates[-possible_dict[action]].to_pile
        possible_dict[action] += 1
        num = 1

        for pile in crane.opposite.from_piles:
            if pile in possible_dict.keys():
                possible_dict[pile] += 1

        while num < self.multi_num:
            same_to_pile = list()
            possible_action = list()
            for pile in possible_dict.keys():
                if len(self.piles[pile].plates) >= possible_dict[pile]:
                    candidate_plate = self.piles[pile].plates[-possible_dict[pile]]
                    if candidate_plate.to_pile == current_to_pile:
                        same_to_pile.append(pile)
                    if not candidate_plate.to_pile in ["cn1", "cn2", "cn3"]:
                        target_coord = self.piles[candidate_plate.to_pile].coord
                        if ((crane.name == "Crane-1" and target_coord[0] <= self.max_x - self.safety_margin) or
                                (crane.name == "Crane-2" and target_coord[0] >= 1 + self.safety_margin)):
                            if (abs(target_coord[0] - self.piles[current_to_pile].coord[0]) <= self.multi_dist
                                    and target_coord[1] == self.piles[current_to_pile].coord[1]):
                                if weight + candidate_plate.w <= self.multi_w:
                                    possible_action.append(pile)

            intersection = list(set(same_to_pile) & set(possible_action))
            if len(intersection) != 0:
                if action in intersection:
                    action = action
                else:
                    action = random.choice(intersection)
            else:
                if len(possible_action) != 0:
                    action = random.choice(possible_action)
                else:
                    break
            from_list.append(action)
            weight += self.piles[action].plates[-possible_dict[action]].w
            current_to_pile = self.piles[action].plates[-possible_dict[action]].to_pile
            possible_dict[action] += 1
            num += 1

        for pile in from_list:
            crane.from_piles.append(pile)
            self.move_list.remove(pile)

    def crane_run(self, crane):
        if len(crane.from_piles) == 0:
            self.waiting_crane = crane
            crane.idle = True
            crane.idle_event = self.env.event()

            waiting_start = self.env.now
            if self.monitor.record_events:
                self.monitor.record(self.env.now, "Waiting Start", crane=crane.name,
                                    location=self.location_mapping[crane.current_coord].name)

            yield crane.idle_event

            waiting_finish = self.env.now
            if self.record_events:
                self.monitor.record(self.env.now, "Waiting Finish", crane=crane.name,
                                    location=self.location_mapping[crane.current_coord].name)
            crane.waiting_time += waiting_finish - waiting_start
        else:
            crane.idle = False

            # Plate loading
            crane.loading = True
            yield self.env.process(self.collision_avoidance(crane))
            crane.loading = False

            # Plate Unloading
            crane.unloading = True
            yield self.env.process(self.collision_avoidance(crane))
            crane.unloading = False

            if (crane.opposite.idle_event is not None) and (not crane.opposite.idle_event.triggered):
                crane.opposite.idle_event.succeed()

        crane.idle = True
        crane.target_coord = (-1.0, -1.0)
        # release a crane
        yield self.cranes.put(crane)

    def collision_avoidance(self, crane):
        self.priority_queue.append(crane.name)

        if crane.loading:
            location_list = crane.from_piles[:]
        elif crane.unloading:
            location_list = crane.to_piles[:]
        else:
            location_list = []

        for location in location_list:
            added_moving_time = 0.0
            if location in self.piles.keys():
                crane.target_coord = self.piles[location].coord
            else:
                crane.target_coord = (self.conveyors[location].coord[0], crane.current_coord[1])

            if crane.opposite.idle:
                num_loop = 1
            else:
                if self.priority_queue.index(crane.name) == 0:
                    num_loop = 1
                else:
                    if crane.opposite.loading:
                        num_loop = len(crane.opposite.from_piles)
                    elif crane.opposite.unloading:
                        num_loop = len(crane.opposite.to_piles)
                    else:
                        num_loop = 1

            cnt = 0
            while cnt < num_loop:
                avoidance, safety_xcoord = self.check_interference(crane)
                if avoidance:
                    crane.safety_xcoord = safety_xcoord
                    opposite_direction = True if np.sign(crane.safety_xcoord - crane.current_coord[0]) \
                                                 != np.sign(crane.target_coord[0] - crane.current_coord[0]) else False
                    moving_time_crane = crane.get_moving_time(to_xcoord=safety_xcoord)
                    moving_time_opposite_crane = crane.opposite.get_moving_time(to_xcoord=crane.opposite.target_coord[0],
                                                                                to_ycoord=crane.opposite.target_coord[1])

                    if self.monitor.record_events:
                        self.monitor.record(self.env.now, "Move_from", crane=crane.name,
                                            location=self.location_mapping[crane.current_coord].name, plate=None)
                    moving_time = yield self.env.process(crane.move(to_xcoord=safety_xcoord))
                    if self.monitor.record_events:
                        self.monitor.record(self.env.now, "Move_to", crane=crane.name,
                                            location=self.location_mapping[crane.current_coord].name, plate=None)
                    crane.safety_xcoord = -1.0

                    if opposite_direction:
                        crane.avoiding_time += moving_time
                        added_moving_time += moving_time
                    else:
                        if crane.loading:
                            crane.empty_travel_time += moving_time

                    if moving_time_opposite_crane > moving_time_crane:
                        avoiding_start = self.env.now
                        if self.monitor.record_events:
                            self.monitor.record(self.env.now, "Avoiding_wait_start", crane=crane.name,
                                                location=self.location_mapping[crane.current_coord].name, plate=None)

                        # crane.avoiding_event = self.env.event()
                        # yield crane.avoiding_event
                        yield self.env.timeout(moving_time_opposite_crane - moving_time_crane)

                        avoiding_finish = self.env.now
                        if self.monitor.record_events:
                            self.monitor.record(self.env.now, "Avoiding_wait_finish", crane=crane.name,
                                                location=self.location_mapping[crane.current_coord].name, plate=None)
                        crane.avoiding_time += avoiding_finish - avoiding_start
                        # crane.avoiding_time += moving_time_opposite_crane - moving_time_crane
                    num_loop += 1
                else:
                    if self.monitor.record_events:
                        self.monitor.record(self.env.now, "Move_from", crane=crane.name,
                                            location=self.location_mapping[crane.current_coord].name, plate=None)
                    moving_time = yield self.env.process(crane.move(to_xcoord=crane.target_coord[0],
                                                                    to_ycoord=crane.target_coord[1]))
                    if self.monitor.record_events:
                        self.monitor.record(self.env.now, "Move_to", crane=crane.name,
                                            location=self.location_mapping[crane.current_coord].name, plate=None)

                    if added_moving_time > 0.0:
                        crane.avoiding_time += added_moving_time

                    if crane.loading:
                        crane.empty_travel_time += (moving_time - added_moving_time)

                    # if (crane.opposite.avoiding_event is not None) and (not crane.opposite.avoiding_event.triggered):
                    #     crane.opposite.avoiding_event.succeed()

                cnt += 1

            assert self.location_mapping[crane.current_coord].name == location
            if crane.loading:
                plate_name = crane.get_plate(self.location_mapping[crane.current_coord])
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Pick_up", crane=crane.name,
                                        location=self.location_mapping[crane.current_coord].name, plate=plate_name)
                crane.from_piles.remove(self.location_mapping[crane.current_coord].name)
                crane.to_piles.insert(0, crane.plates[-1].to_pile)
            else:
                plate_name = crane.put_plate(self.location_mapping[crane.current_coord])
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Put_down", crane=crane.name,
                                        location=self.location_mapping[crane.current_coord].name, plate=plate_name)
                crane.to_piles.remove(self.location_mapping[crane.current_coord].name)
                self.num_plates_cum += 1

        self.priority_queue.remove(crane.name)

    def check_interference(self, crane):
        if crane.opposite.idle or self.priority_queue.index(crane.name) == 0:
            avoidance = False
            safety_xcoord = None
        else:
            dx = crane.target_coord[0] - crane.current_coord[0]
            dy = crane.target_coord[1] - crane.current_coord[1]
            direction_crane = np.sign(dx)
            moving_time_crane = max(abs(dx) / crane.x_velocity, abs(dy) / crane.y_velocity)

            trajectory_opposite_crane = []
            if crane.opposite.loading:
                location_list = crane.opposite.from_piles
            elif crane.opposite.unloading:
                location_list = crane.opposite.to_piles
            else:
                location_list = []

            current_coord_opposite_crane = crane.opposite.current_coord
            for location in location_list:
                if location in self.piles.keys():
                    dx = self.piles[location].coord[0] - current_coord_opposite_crane[0]
                    dy = self.piles[location].coord[1] - current_coord_opposite_crane[1]
                else:
                    dx = self.conveyors[location].coord[0] - current_coord_opposite_crane[0]
                    dy = 0

                direction_opposite_crane = np.sign(dx)
                moving_time_opposite_crane = max(abs(dx) / crane.opposite.x_velocity, abs(dy) / crane.opposite.y_velocity)
                if location in self.piles.keys():
                    coord_opposite_crane = self.piles[location].coord
                else:
                    coord_opposite_crane = (self.conveyors[location].coord[0], current_coord_opposite_crane[1])
                trajectory_opposite_crane.append((direction_opposite_crane, moving_time_opposite_crane, coord_opposite_crane))
                current_coord_opposite_crane = coord_opposite_crane

            current_coord_opposite_crane = crane.opposite.current_coord
            moving_time_cum = 0.0
            avoidance = False
            safety_xcoord = None
            for i, move in enumerate(trajectory_opposite_crane):
                if moving_time_crane <= move[1] + moving_time_cum:
                    min_moving_time = moving_time_crane
                    min_crane = crane.name
                else:
                    min_moving_time = move[1] + moving_time_cum
                    min_crane = crane.opposite.name

                xcoord_crane = crane.current_coord[0] + min_moving_time * crane.x_velocity * direction_crane
                xcoord_opposite_crane = (current_coord_opposite_crane[0] + (min_moving_time - moving_time_cum)
                                         * crane.opposite.x_velocity * move[0])

                if (crane.name == 'Crane-1' and xcoord_crane > xcoord_opposite_crane - self.safety_margin) \
                        or (crane.name == 'Crane-2' and xcoord_crane < xcoord_opposite_crane + self.safety_margin):
                    avoidance = True
                    if crane.name == 'Crane-1':
                        safety_xcoord = min([temp[2][0] for temp in trajectory_opposite_crane[i:]]) - self.safety_margin
                    else:
                        safety_xcoord = max([temp[2][0] for temp in trajectory_opposite_crane[i:]]) + self.safety_margin
                    break
                else:
                    if crane.name == min_crane:
                        check_same_pile = ((crane.loading and crane.opposite.loading)
                                           or (crane.loading and crane.opposite.unloading)
                                           or (crane.unloading and crane.opposite.unloading))
                        same_pile = False
                        if check_same_pile:
                            for temp in trajectory_opposite_crane[i:]:
                                if crane.target_coord[0] == temp[2][0] and crane.target_coord[1] == temp[2][1]:
                                    same_pile = True

                        if same_pile:
                            avoidance = True
                            if crane.name == "Crane-1":
                                safety_xcoord = crane.target_coord[0] - self.safety_margin
                            else:
                                safety_xcoord = crane.target_coord[0] + self.safety_margin
                        else:
                            avoidance = False
                            safety_xcoord = None
                        break
                    else:
                        if len(trajectory_opposite_crane) == i + 1:
                            avoidance = False
                            safety_xcoord = None
                        else:
                            current_coord_opposite_crane = move[2]
                            moving_time_cum = min_moving_time

        return avoidance, safety_xcoord

    # 이동 대상 파일에 선별 작업을 수행해야 할 다른 강재가 적치되어 있어서
    # 당장 적치되어 있는 강재를 이동하는 것이 불가능한 파일을 식별
    def check_blocked_piles(self):
        # 시뮬레이션 시작 전 선택 불가한 파일 체크
        if self.last_action == "Waiting":
            pass
        elif self.last_action is None:
            for from_pile in self.df_reshuffle["pileno"].unique():
                to_pile = self.piles[from_pile].plates[-1].to_pile
                if to_pile in self.move_list:
                    self.blocked_piles.append(from_pile)
                    self.piles[to_pile].blocking.append(from_pile)
        # 시뮬레이션 도중에 에이전트가 행동을 선택할 때마다 추가적으로 선택 불가한 파일이 발생하는 지 체크
        else:
            if not self.last_action in self.move_list:
                while self.piles[self.last_action].blocking:
                    self.blocked_piles.remove(self.piles[self.last_action].blocking.pop())
            else:
                to_pile = self.piles[self.last_action].plates[-1].to_pile
                if to_pile in self.move_list:
                    self.blocked_piles.append(self.last_action)
                    self.piles[to_pile].blocking.append(self.last_action)

    def release(self, conveyor):
        while self.move_list:
            try:
                # IAT = random.expovariate(conveyor.IAT)
                IAT = np.random.geometric(conveyor.IAT)
                yield self.env.timeout(IAT)
                if self.monitor.record_events:
                    self.monitor.record(self.env.now, "Release", crane=None, location=conveyor.name, plate=None)

                # request a crane
                if self.waiting_crane is not None:
                    if self.waiting_crane.name == 'Crane-1' and conveyor.name == "cn3":
                        crane = yield self.cranes.get(priority=1, filt=lambda x: x.name == "Crane-2")
                    else:
                        remaining_time = max(self.waiting_crane.move_until - self.env.now, 0)
                        if remaining_time > 0:
                            yield self.env.timeout(remaining_time)
                        if not self.waiting_crane.idle_event.triggered:
                            self.waiting_crane.idle_event.succeed()
                        if conveyor.name == "cn3":
                            crane = yield self.cranes.get(priority=1, filt=lambda x: x.name == "Crane-2")
                        else:
                            crane = yield self.cranes.get(priority=1, filt=lambda x: True)
                    crane.opposite.update_location(self.env.now)
                    crane.update_location(self.env.now, on_pile=True)
                else:
                    if conveyor.name == "cn3":
                        crane = yield self.cranes.get(priority=1, filt=lambda x: x.name == "Crane-2")
                    else:
                        crane = yield self.cranes.get(priority=1, filt=lambda x: True)

                # 출고 작업을 수행할 강재가 적치되어 있는 파일 리스트 생성
                candidates = []
                for from_pile_name in self.df_retrieval["pileno"].unique():
                    cnt = crane.opposite.from_piles.count(from_pile_name)
                    if cnt > 0:
                        plates = self.piles[from_pile_name].plates[:-1-cnt]
                    else:
                        plates = self.piles[from_pile_name].plates[:]
                    if len(plates) > 0:
                        temp = self.df_retrieval[self.df_retrieval["pileno"] == from_pile_name]
                        if conveyor.name in temp["topile"].unique():
                            candidates.append(from_pile_name)

                # 생성된 파일 리스트에서 랜덤하게 파일을 하나 선택하고 해당 파일에 적치된 강재에 대한 출고 작업 수행
                if len(candidates) > 0:
                    crane.idle = False

                    from_pile = random.choice(candidates)
                    crane.from_piles.append(from_pile)

                    # Plate Loading
                    crane.loading = True
                    yield self.env.process(self.collision_avoidance(crane))
                    crane.loading = False

                    # Plate Unloading
                    crane.unloading = True
                    yield self.env.process(self.collision_avoidance(crane))
                    crane.unloading = False

                    if (crane.opposite.idle_event is not None) and (not crane.opposite.idle_event.triggered):
                        crane.opposite.idle_event.succeed()

                crane.idle = True
                crane.target_coord = (-1.0, -1.0)

                # release a crane
                yield self.cranes.put(crane)

            except simpy.Interrupt:
                pass