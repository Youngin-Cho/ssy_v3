import os
import time
import json
import string

import torch
# import vessl

from environment.env import *
from agent.network import *
from agent.heuristics import *
from cfg_test import *

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    cfg = get_cfg()
    # vessl.init(organization="snu-eng-dgx", project="quay", hp=cfg)

    model_path = cfg.model_path
    param_path = cfg.param_path
    # data_dir = cfg.data_dir
    # log_dir = cfg.log_dir

    # data_dir = ["./input/data/test/scalability_test/storage_plan/10-10-10/",
    #             "./input/data/test/scalability_test/storage_plan/15-10-10/",
    #             "./input/data/test/scalability_test/storage_plan/20-10-10/",
    #             "./input/data/test/scalability_test/reshuffling_plan/5-15-10/",
    #             "./input/data/test/scalability_test/reshuffling_plan/5-20-10/",
    #             "./input/data/test/scalability_test/reshuffling_plan/5-25-10/",
    #             "./input/data/test/scalability_test/reshuffling_plan/5-10-15/",
    #             "./input/data/test/scalability_test/reshuffling_plan/5-10-20/",
    #             "./input/data/test/scalability_test/reshuffling_plan/5-10-25/"]
    # log_dir = ["./output/test/scalability_test/storage_plan/10-10-10/",
    #             "./output/test/scalability_test/storage_plan/15-10-10/",
    #             "./output/test/scalability_test/storage_plan/20-10-10/",
    #             "./output/test/scalability_test/reshuffling_plan/5-15-10/",
    #             "./output/test/scalability_test/reshuffling_plan/5-20-10/",
    #             "./output/test/scalability_test/reshuffling_plan/5-25-10/",
    #             "./output/test/scalability_test/reshuffling_plan/5-10-15/",
    #             "./output/test/scalability_test/reshuffling_plan/5-10-20/",
    #             "./output/test/scalability_test/reshuffling_plan/5-10-25/"]

    data_dir = ["./input/data/test/basic_test/5-10-10/",]
    log_dir = ["./output/test/basic_test/iat/5-10-10/up10/"]

    with open(param_path, 'r') as f:
        parameters = json.load(f)

    algorithm = cfg.algorithm
    use_gnn = bool(cfg.use_gnn)
    random_seed = cfg.random_seed
    record_events = bool(cfg.record_events)

    n_bays_in_area1 = cfg.n_bays_in_area1
    n_bays_in_area2 = cfg.n_bays_in_area2
    n_bays_in_area3 = cfg.n_bays_in_area3
    n_bays_in_area4 = cfg.n_bays_in_area4
    n_bays_in_area5 = cfg.n_bays_in_area5
    n_bays_in_area6 = cfg.n_bays_in_area6

    max_x = n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6 + 4
    max_y = cfg.n_rows
    row_range = (string.ascii_uppercase[0], string.ascii_uppercase[cfg.n_rows - 1])
    bay_range = (1, n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6)
    input_points = (1,)
    output_points = (1 + n_bays_in_area1 + n_bays_in_area2 + 1,
                     1 + n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + 2,
                     1 + n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6 + 3)
    working_crane_ids = tuple()
    if bool(cfg.is_crane1_working):
        working_crane_ids = working_crane_ids + ("Crane-1",)
    if bool(cfg.is_crane2_working):
        working_crane_ids = working_crane_ids + ("Crane-2",)
    safety_margin = cfg.safety_margin

    multi_num = cfg.multi_num
    multi_w = cfg.multi_w
    multi_dis = cfg.multi_dis

    for log_dir_temp in log_dir:
        if not os.path.exists(log_dir_temp):
            os.makedirs(log_dir_temp)

    for data_dir_temp, log_dir_temp in zip(data_dir, log_dir):
        test_paths = os.listdir(data_dir_temp)

        index = ["P%d" % i for i in range(1, len(test_paths) + 1)] + ["avg"]
        # columns = ["RL", "SETT", "NCR", "TDD", "TDT", "RAND"] if algorithm == "ALL" else [algorithm]
        # columns = ["RL"]
        columns = ["SETT", "NCR", "TDD", "TDT", "RAND"]
        df_makespan = pd.DataFrame(index=index, columns=columns)
        df_empty_travel_time_1 = pd.DataFrame(index=index, columns=columns)
        df_avoiding_time_1 = pd.DataFrame(index=index, columns=columns)
        df_idle_time_1 = pd.DataFrame(index=index, columns=columns)
        df_empty_travel_time_2 = pd.DataFrame(index=index, columns=columns)
        df_avoiding_time_2 = pd.DataFrame(index=index, columns=columns)
        df_idle_time_2 = pd.DataFrame(index=index, columns=columns)
        df_computing_time = pd.DataFrame(index=index, columns=columns)

        for name in columns:
            progress = 0
            list_makespan = []
            list_empty_travel_time_1 = []
            list_avoiding_time_1 = []
            list_idle_time_1 = []
            list_empty_travel_time_2 = []
            list_avoiding_time_2 = []
            list_idle_time_2 = []
            list_computing_time = []

            for prob, path in zip(index, test_paths):
                random.seed(random_seed)

                makespan = 0.0
                empty_travel_time = 0.0
                avoiding_time = 0.0
                computing_time = 0.0

                if name == "RL":
                    env = SteelStockYard(data_dir_temp + path, look_ahead=parameters["look_ahead"],
                                         max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                         input_points=input_points, output_points=output_points,
                                         working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                                         multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                                         algorithm="RL", record_events=record_events)
                else:
                    env = SteelStockYard(data_dir_temp + path, look_ahead=parameters["look_ahead"],
                                         max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                         input_points=input_points, output_points=output_points,
                                         working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                                         multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                                         algorithm=name, record_events=record_events)

                if name == "RL":
                    device = torch.device("cpu")
                    agent = Scheduler(env.meta_data, env.state_size, env.num_nodes,
                                      int(parameters['embed_dim']),
                                      int(parameters['num_heads']),
                                      int(parameters['num_HGT_layers']),
                                      int(parameters['num_actor_layers']),
                                      int(parameters['num_critic_layers']),
                                      use_gnn=use_gnn).to(torch.device('cpu'))
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    agent.load_state_dict(checkpoint['model_state_dict'])

                start = time.time()
                state, mask, crane_id = env.reset()
                done = False

                while not done:
                    if name == "RL":
                        action, _, _ = agent.act(state, mask, crane_id, greedy=False)
                    elif name == "SETT":
                        action = SETT(state, mask)
                    elif name == "NCR":
                        action = NCR(state, mask)
                    elif name == "TDD":
                        action = TDD(state, mask)
                    elif name == "TDT":
                        action = TDT(state, mask)
                    else:
                        action = RAND(state, mask)

                    next_state, reward, done, mask, next_crane_id = env.step(action)
                    state = next_state
                    crane_id = next_crane_id

                    if done:
                        finish = time.time()
                        makespan = env.model.env.now
                        for crane_name in env.model.reward_info.keys():
                            if crane_name == "Crane-1":
                                empty_travel_time_1 = env.model.reward_info[crane_name]["Empty Travel Time Cumulative"]
                                avoiding_time_1 = env.model.reward_info[crane_name]["Avoiding Time Cumulative"]
                                idle_time_1 = env.model.reward_info[crane_name]["Idle Time Cumulative"]
                            elif crane_name == "Crane-2":
                                empty_travel_time_2 = env.model.reward_info[crane_name]["Empty Travel Time Cumulative"]
                                avoiding_time_2 = env.model.reward_info[crane_name]["Avoiding Time Cumulative"]
                                idle_time_2 = env.model.reward_info[crane_name]["Idle Time Cumulative"]
                        computing_time = finish - start
                        # env.get_logs(log_dir + "sim_%s.csv" % name)
                        break

                list_makespan.append(makespan)
                list_empty_travel_time_1.append(empty_travel_time_1)
                list_avoiding_time_1.append(avoiding_time_1)
                list_idle_time_1.append(idle_time_1)
                list_empty_travel_time_2.append(empty_travel_time_2)
                list_avoiding_time_2.append(avoiding_time_2)
                list_idle_time_2.append(idle_time_2)
                list_computing_time.append(computing_time)

                progress += 1
                print("%d/%d test for %s done" % (progress, len(index) - 1, name))

            df_makespan[name] = list_makespan + [sum(list_makespan) / len(list_makespan)]
            df_empty_travel_time_1[name] = list_empty_travel_time_1 + [sum(list_empty_travel_time_1) / len(list_empty_travel_time_1)]
            df_avoiding_time_1[name] = list_avoiding_time_1 + [sum(list_avoiding_time_1) / len(list_avoiding_time_1)]
            df_idle_time_1[name] = list_idle_time_1 + [sum(list_idle_time_1) / len(list_idle_time_1)]
            df_empty_travel_time_2[name] = list_empty_travel_time_2 + [sum(list_empty_travel_time_2) / len(list_empty_travel_time_2)]
            df_avoiding_time_2[name] = list_avoiding_time_2 + [sum(list_avoiding_time_2) / len(list_avoiding_time_2)]
            df_idle_time_2[name] = list_idle_time_2 + [sum(list_idle_time_2) / len(list_idle_time_2)]
            df_computing_time[name] = list_computing_time + [sum(list_computing_time) / len(list_computing_time)]
            print("==========test for %s finished==========" % name)

        writer = pd.ExcelWriter(log_dir_temp + '(Heuristics) test_results.xlsx')
        df_makespan.to_excel(writer, sheet_name="makespan")
        df_empty_travel_time_1.to_excel(writer, sheet_name="empty_travel_time_1")
        df_avoiding_time_1.to_excel(writer, sheet_name="avoiding_time_1")
        df_idle_time_1.to_excel(writer, sheet_name="idle_time_1")
        df_empty_travel_time_2.to_excel(writer, sheet_name="empty_travel_time_2")
        df_avoiding_time_2.to_excel(writer, sheet_name="avoiding_time_2")
        df_idle_time_2.to_excel(writer, sheet_name="idle_time_2")
        df_computing_time.to_excel(writer, sheet_name="computing_time")
        writer.close()