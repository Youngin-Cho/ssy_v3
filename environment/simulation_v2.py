import simpy
import random
import numpy as np
import pandas as pd

from collections import OrderedDict
from utilities import get_coord, get_moving_time


# random.seed(42)

class PriorityGet(simpy.resources.base.Get):
    def __init__(self, resource, priority=10, preempt=True):
        self.priority = priority
        self.preempt = preempt
        self.time = resource._env.now
        self.usage_since = None
        self.key = (self.priority, self.time, not self.preempt)
        super().__init__(resource)


class PriorityBaseStore(simpy.resources.store.Store):

    GetQueue = simpy.resources.resource.SortedQueue
    get = simpy.core.BoundClass(PriorityGet)


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


class Crane:
    def __init__(self, env, name, location, safety_margin):
        self.env = env
        self.name = name
        self.current_location = location
        self.safety_margin = safety_margin

        self.w_limit = 8.0
        self.current_coord = location.coord
        self.target_coord = (-1.0, -1.0)
        self.plates = list()

        self.opposite = None
        self.idle = False
        self.idle_event = self.env.event()

        self.moving = True
        self.x_velocity = 0.5
        self.y_velocity = 1
        self.update_time = 0

    def get_plate(self, pile):
        plate = pile.plates.pop()
        self.plates.append(plate)
        return plate.name

    def put_plate(self, pile):
        plate = self.plates.pop()
        pile.plates.append(plate)
        return plate.name

    def move(self, to_xcoord=None, to_ycoord=None):
        self.moving = True

        moving_time = self.get_moving_time(to_xcoord, to_ycoord)
        yield self.env.timeout(moving_time)

        self.opposite.update_location(self.env.now)
        self.update_location(self.env.now)

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
        return moving_time

    def update_location(self, time):
        x_coord = self.current_coord[0]
        y_coord = self.current_coord[1]

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
                time_elapsed = time - self.update_time
                x_coord = x_coord + time_elapsed * self.x_velocity * np.sign(self.target_coord[0] - x_coord)
                y_coord = y_coord + time_elapsed * self.y_velocity * np.sign(self.target_coord[1] - y_coord)
                x_coord = np.clip(x_coord, 1, 44)
                y_coord = np.clip(y_coord, 0, 1)

        self.update_time = time
        self.current_coord = (x_coord, y_coord)


class Conveyor:
    def __init__(self, name, coord, IAT):
        self.name = name
        self.coord = [coord]
        self.IAT = IAT

        self.plates = list()


class Monitor:
    def __init__(self):
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
    def __init__(self, df_storage, df_reshuffle, df_retrieval, bays=("A", "B"), safety_margin=5):
        self.df_storage = df_storage
        self.df_reshuffle = df_reshuffle
        self.df_retrieval = df_retrieval
        self.bays = bays
        self.safety_margin = safety_margin

        self.env, self.piles, self.conveyors, self.cranes, self.monitor = self._modeling()
        self.move_list = list(df_storage["pileno"].values) + list(df_reshuffle["pileno"].values)
        self.blocked_piles = list()
        self.last_action = None
        self.crane_info = {crane.name: {"Current Coord": crane.current_coord,
                                        "Target Coord": (-1.0, -1.0),
                                        "Status": 0} for crane in self.cranes.items}
        self.idle_time = {crane.name: 0 for crane in self.cranes.items}  # empty travel 단계의 이동거리
        self.idle_time_cum = {crane.name: 0 for crane in self.cranes.items}

        self.pile_info = {tuple(pile.coord): pile for pile in self.piles.values()} # coord를 통해 pile 호출
        for cn, coord in self.conveyors.items():
            self.pile_info[coord] = cn

        self.decision_time = False
        self.crane_in_decision = None
        self.do_action = self.env.event()

        self.action = self.env.process(self.run())
        # if df_retrieval is not None:
        #     self.action_conveyor = [self.env.process(self.release(cn)) for cn in self.conveyors.values()]

    def _modeling(self):
        env = simpy.Environment()

        pile_list = [row_id + str(col_id).rjust(2, '0') for row_id in self.bays for col_id in range(0, 41)]
        piles = OrderedDict({name: Pile(name, get_coord(name)) for name in pile_list})

        conveyors = OrderedDict()
        conveyors['cn1'] = Conveyor('cn1', 23, 0.01)
        conveyors['cn2'] = Conveyor('cn2', 27, 0.01)
        conveyors['cn3'] = Conveyor('cn3', 44, 0.0005)

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

        cranes = PriorityBaseStore(env)

        crane1 = Crane(env, 'Crane-1', piles[self.bays[0] + "01"], self.safety_margin)
        crane2 = Crane(env, 'Crane-2', piles[self.bays[0] + "40"], self.safety_margin)

        crane1.opposite = crane2
        crane2.opposite = crane1

        cranes.put(crane1)
        cranes.put(crane2)

        monitor = Monitor()

        return env, piles, conveyors, cranes, monitor

    def run(self):
        while self.move_list:
            self.check_blocked_piles()

            # request a crane
            crane = yield self.cranes.get(priority=3)
            self.crane_in_decision = int(crane.name.split("-")[-1]) - 1

            self.crane_info[crane.name]["Current Coord"] = crane.current_coord
            self.crane_info[crane.name]["Target Coord"] = crane.target_coord

            self.crane_info[crane.opposite.name]["Current Coord"] = crane.opposite.current_coord
            self.crane_info[crane.opposite.name]["Target Coord"] = crane.opposite.target_coord

            # 행동 선택을 위한 이벤트 생성
            self.decision_time = True
            self.do_action = self.env.event()
            action = yield self.do_action
            self.last_action = action
            self.do_action = None

            if action != "Waiting":
                self.move_list.remove(self.piles[action].name)

            self.env.process(self.crane_run(crane, action))

        # for action in self.action_conveyor:
        #     action.interrupt()

    def crane_run(self, crane, action):
        if action == "Waiting":
            crane.idle = True
            crane.idle_event = self.env.event()

            waiting_start = self.env.now
            self.monitor.record(self.env.now, "Waiting Start", crane=crane.name, location=crane.current_location.name)

            yield crane.idle_event

            waiting_finish = self.env.now
            self.monitor.record(self.env.now, "Waiting Finish", crane=crane.name, location=crane.current_location.name)
        else:
            crane.idle = False
            from_pile = self.piles[action]
            to_pile = self.piles[from_pile.plates[-1].to_pile]

            # identify the current job
            if "00" in action:
                tag = "Storage"
            else:
                tag = "Reshuflle"

            # empty travel
            crane.target_location = from_pile
            crane.target_coord = from_pile.coord
            yield self.env.process(self.collision_avoidance(crane, tag))

            # pick-up
            plate_name = crane.get_plate(from_pile)
            self.monitor.record(self.env.now, "Pick_up", crane=crane.name,
                                location=crane.current_location.name, plate=plate_name, tag=tag)

            # full travel
            crane.target_location = to_pile
            crane.target_coord = to_pile.coord
            yield self.env.process(self.collision_avoidance(crane, tag))

            # drop-off
            plate_name = crane.put_plate(to_pile)
            self.monitor.record(self.env.now, "Put_down", crane=crane.name,
                                location=crane.current_location.name, plate=plate_name, tag=tag)

            crane.target_location = None
            crane.target_coord = (-1.0, -1.0)

            if crane.opposite.idle:
                crane.opposite.idle_event.succeed()

        # release a crane
        yield self.cranes.put(crane)

    def collision_avoidance(self, crane, tag):
        avoidance, safety_xcoord = self.check_interference(crane)
        if avoidance:
            moving_time_crane = crane.get_moving_time(to_xcoord=safety_xcoord)
            moving_time_opposite_crane = crane.opposite.get_moving_time(to_xcoord=crane.opposite.target_coord[0],
                                                                        to_ycoord=crane.opposite.target_coord[1])

            self.monitor.record(self.env.now, "Move_from", crane=crane.name,
                                location=crane.current_location.name, plate=None, tag=tag)
            yield self.env.process(crane.move(to_xcoord=safety_xcoord))
            self.monitor.record(self.env.now, "Move_to", crane=crane.name,
                                location=crane.current_location.name, plate=None, tag=tag)

            self.monitor.record(self.env.now, "Move_from", crane=crane.name,
                                location=crane.current_location.name, plate=None, tag=tag)
            yield self.env.timeout(moving_time_opposite_crane - moving_time_crane)
            yield self.env.process(crane.move(to_xcoord=crane.target_coord[0],
                                              to_ycoord=crane.target_coord[1]))
            self.monitor.record(self.env.now, "Move_to", crane=crane.name,
                                location=crane.current_location.name, plate=None, tag=tag)
        else:
            self.monitor.record(self.env.now, "Move_from", crane=crane.name,
                                location=crane.current_location.name, plate=None, tag=tag)
            moving_time_crane = yield self.env.process(crane.move(to_xcoord=crane.target_coord[0],
                                                                  to_ycoord=crane.target_coord[1]))
            self.monitor.record(self.env.now, "Move_to", crane=crane.name,
                                location=crane.current_location.name, plate=None, tag=tag)

    def check_interference(self, crane):
        if crane.opposite.idle:
            avoidance = False
            safety_xcoord = None
        else:
            target_xcoord_crane = crane.target_coord[0]
            target_xcoord_opposite_crane = crane.opposite.target_coord[0]

            if (crane.name == 'Crane-1' and target_xcoord_crane >= target_xcoord_opposite_crane) \
                or (crane.name == 'Crane-2' and target_xcoord_crane <= target_xcoord_opposite_crane):

                if crane.name == 'Crane-1':
                    safety_xcoord = target_xcoord_opposite_crane - self.safety_margin
                else:
                    safety_xcoord = target_xcoord_opposite_crane + self.safety_margin

                moving_time_crane = crane.get_moving_time(to_xcoord=safety_xcoord)
                moving_time_opposite_crane = crane.opposite.get_moving_time(to_xcoord=crane.opposite.target_coord[0],
                                                                            to_ycoord=crane.opposite.target_coord[1])

                if moving_time_crane >= moving_time_opposite_crane:
                    avoidance = False
                    safety_xcoord = None
                else:
                    avoidance = True
            else:
                avoidance = False
                safety_xcoord = None

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
                IAT = random.expovariate(conveyor.IAT)
                yield self.env.timeout(IAT)
                self.monitor.record(self.env.now, "Release", crane=None, location=conveyor.name, plate=None, tag=None)

                # request a crane
                crane = yield self.cranes.get(priority=1)

                # 출고 작업을 수행할 강재가 적치되어 있는 파일 리스트 생성
                candidates = [i for i in self.df_retrieval["pileno"].unique()
                              if conveyor.name in self.df_retrieval["topile"] and len(self.piles[i].plates) > 0]

                # 생성된 파일 리스트에서 랜덤하게 파일을 하나 선택하고 해당 파일에 적치된 강재에 대한 출고 작업 수행
                if len(candidates) > 0:
                    from_pile_name = random.choice(candidates)
                    from_pile = self.piles[from_pile_name]

                    # empty travel
                    self.monitor.record(self.env.now, "Move_from", crane=crane.name,
                                        location=crane.current_location.name, plate=None, tag="Retrieval")
                    distance = yield self.env.process(crane.move(from_pile))
                    self.monitor.record(self.env.now, "Move_to", crane=crane.name,
                                        location=crane.current_location.name, plate=None, tag="Retrieval")

                    # pick-up
                    plate_name = crane.get_plate(from_pile)
                    self.monitor.record(self.env.now, "Pick_up", crane=crane.name,
                                        location=crane.current_location.name, plate=plate_name, tag="Retrieval")

                    # full travel
                    self.monitor.record(self.env.now, "Move_from", crane=crane.name,
                                        location=crane.current_location.name, plate=plate_name, tag="Retrieval")
                    yield self.env.process(crane.move(conveyor))
                    self.monitor.record(self.env.now, "Move_to", crane=crane.name,
                                        location=crane.current_location.name, plate=plate_name, tag="Retrieval")

                    # drop-off
                    plate_name = crane.put_plate(conveyor)
                    self.monitor.record(self.env.now, "Put_down", crane=crane.name,
                                        location=crane.current_location.name, plate=plate_name, tag="Retrieval")

                    self.crane_info[crane.name]["Current Coord"] = crane.current_coord
                    self.crane_info[crane.name]["Target Coord"] = (-1.0, -1.0)
                    self.crane_info[crane.name]["Status"] = 0

                # release a crane
                yield self.cranes.put(crane)

            except simpy.Interrupt:
                pass