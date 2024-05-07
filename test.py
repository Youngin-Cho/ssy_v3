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
    data_dir = cfg.data_dir
    log_dir = cfg.log_dir

    with open(param_path, 'r') as f:
        parameters = json.load(f)

    algorithm = cfg.algorithm
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

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    test_paths = os.listdir(data_dir)
    index = ["P%d" % i for i in range(1, len(test_paths) + 1)] + ["avg"]
    columns = ["RL", "SD", "MA", "MX", "SRF", "SRT", "RAND"] if algorithm == "ALL" else [algorithm]
    df_makespan = pd.DataFrame(index=index, columns=columns)
    df_empty_travel_time = pd.DataFrame(index=index, columns=columns)
    df_avoiding_time = pd.DataFrame(index=index, columns=columns)
    df_computing_time = pd.DataFrame(index=index, columns=columns)

    for name in columns:
        progress = 0
        list_makespan = []
        list_empty_travel_time = []
        list_avoiding_time = []
        list_computing_time = []

        for prob, path in zip(index, test_paths):
            random.seed(random_seed)

            makespan = 0.0
            empty_travel_time = 0.0
            avoiding_time = 0.0
            computing_time = 0.0

            if name == "RL":
                env = SteelStockYard(data_dir + path, look_ahead=parameters["look_ahead"],
                                     max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                     input_points=input_points, output_points=output_points,
                                     working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                                     multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                                     rl=True, record_events=record_events)
            else:
                env = SteelStockYard(data_dir + path, look_ahead=parameters["look_ahead"],
                                     max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                     input_points=input_points, output_points=output_points,
                                     working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                                     multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                                     rl=False, record_events=record_events)

            if name == "RL":
                device = torch.device("cpu")
                agent = Scheduler(env.meta_data, env.state_size, env.num_nodes,
                                  int(parameters['embed_dim']),
                                  int(parameters['num_heads']),
                                  int(parameters['num_HGT_layers']),
                                  int(parameters['num_actor_layers']),
                                  int(parameters['num_critic_layers'])).to(torch.device('cpu'))
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                agent.load_state_dict(checkpoint['model_state_dict'])

            start = time.time()
            state, mask, crane_id = env.reset()
            done = False

            while not done:
                if name == "RL":
                    action, _, _ = agent.act(state, mask, crane_id, greedy=False)
                elif name == "SD":
                    action = shortest_distance(state, mask)
                elif name == "MA":
                    action = minimize_avoiding_time(state, mask)
                elif name == "MX":
                    action = mixed_heruistic(state, mask)
                elif name == "SRF":
                    action = separate_regions_by_from_pile(state, mask)
                elif name == "SRT":
                    action = separate_regions_by_to_pile(state, mask)
                else:
                    action = random_selection(state, mask)

                next_state, reward, done, mask, next_crane_id = env.step(action)
                state = next_state
                crane_id = next_crane_id

                if done:
                    finish = time.time()
                    makespan = env.model.env.now
                    for crane_name in env.model.reward_info.keys():
                        empty_travel_time += env.model.reward_info[crane_name]["Empty Travel Time Cumulative"]
                        avoiding_time += env.model.reward_info[crane_name]["Avoiding Time Cumulative"]
                    computing_time = finish - start
                    break

            list_makespan.append(makespan)
            list_empty_travel_time.append(empty_travel_time)
            list_avoiding_time.append(avoiding_time)
            list_computing_time.append(computing_time)

            progress += 1
            print("%d/%d test for %s done" % (progress, len(index) - 1, name))

        df_makespan[name] = list_makespan + [sum(list_makespan) / len(list_makespan)]
        df_empty_travel_time[name] = list_empty_travel_time + [
            sum(list_empty_travel_time) / len(list_empty_travel_time)]
        df_avoiding_time[name] = list_avoiding_time + [sum(list_avoiding_time) / len(list_avoiding_time)]
        df_computing_time[name] = list_computing_time + [sum(list_computing_time) / len(list_computing_time)]
        print("==========test for %s finished==========" % name)

    writer = pd.ExcelWriter(log_dir + 'test_results.xlsx')
    df_makespan.to_excel(writer, sheet_name="makespan")
    df_empty_travel_time.to_excel(writer, sheet_name="empty_travel_time")
    df_avoiding_time.to_excel(writer, sheet_name="avoiding_time")
    df_computing_time.to_excel(writer, sheet_name="computing_time")
    writer.close()