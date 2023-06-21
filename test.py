import os
import vessl
import time
import torch
import random
import numpy as np
import pandas as pd

from agent.iqn import *
from benchmark.heuristics import *
from environment.data import *
from environment.env import *
from cfg_test import *


if __name__ == "__main__":
    cfg = get_cfg()
    vessl.init(organization="snu-eng-dgx", project="ssy", hp=cfg)

    model_path = cfg.model_path
    data_folder = cfg.data_folder

    algorithm = cfg.algorithm
    iteration = cfg.iteration
    random_seed = cfg.random_seed

    safety_margin = cfg.safety_margin

    # simulation_dir = './output/test/simulation/case1/case1-1'
    # if not os.path.exists(simulation_dir):
    #     os.makedirs(simulation_dir)

    log_dir = '/output/case_study/case1/case1-1/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    test_paths = os.listdir(data_folder)
    index = [str(i) for i in range(1, iteration + 1)] + ["avg"]
    columns = ["P%d" % i for i in range(1, len(test_paths) + 1)]

    random.seed(random_seed)
    progress = 0
    df_makespan = pd.DataFrame(index=index, columns=columns)
    df_empty_travel_time = pd.DataFrame(index=index, columns=columns)
    df_avoiding_time = pd.DataFrame(index=index, columns=columns)
    df_computing_time = pd.DataFrame(index=index, columns=columns)

    for prob, path in zip(columns, test_paths):
        list_makespan = []
        list_empty_travel_time = []
        list_avoiding_time = []
        list_computing_time = []

        for j in range(iteration):
            makespan = 0.0
            empty_travel_time = 0.0
            avoiding_time = 0.0
            computing_time = 0.0

            df_storage = pd.read_excel(data_folder + path, sheet_name="storage", engine="openpyxl")
            df_reshuffle = pd.read_excel(data_folder + path, sheet_name="reshuffle", engine="openpyxl")
            df_retrieval = pd.read_excel(data_folder + path, sheet_name="retrieval", engine="openpyxl")

            env = SteelStockYard(df_storage=df_storage, df_reshuffle=df_reshuffle, df_retrieval=df_retrieval,
                                 safety_margin=safety_margin)

            if algorithm == "RL":
                device = torch.device("cpu")
                agent = Agent(env.state_size, env.action_size, env.meta_data, n_units=cfg.n_units)
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
            elif algorithm == "SD":
                agent = shortest_distance
            elif algorithm == "LD":
                agent = longest_distance
            elif algorithm == "MA":
                agent = minimize_avoiding_time
            else:
                agent = random_selection

            start = time.time()
            state, info = env.reset()
            crane_in_decision = info["crane_id"]
            done = False

            while not done:
                possible_actions = env.get_possible_actions()
                if algorithm == "RL":
                    action = agent.get_action([state], [possible_actions], eps=0.0,
                                              noisy=False, crane_id=env.crane_in_decision)[0]
                elif algorithm == "SD":
                    action = agent(state, possible_actions, crane_id=crane_in_decision)
                elif algorithm == "LD":
                    action = agent(state, possible_actions, crane_id=crane_in_decision)
                elif algorithm == "MA":
                    action = agent(state, possible_actions, crane_id=crane_in_decision)
                else:
                    action = agent(possible_actions)
                next_state, reward, done, info = env.step(action)
                state = next_state
                crane_in_decision = info["crane_id"]

                if done:
                    finish = time.time()
                    # log = env.get_logs(simulation_dir + 'event_log_{0}_{1}_{2}.csv'.format(name, prob, j))
                    log = env.get_logs()
                    makespan = log["Time"].max()
                    for crane_name in env.model.reward_info.keys():
                        empty_travel_time += env.model.reward_info[crane_name]["Empty Travel Time Cumulative"]
                        avoiding_time += env.model.reward_info[crane_name]["Avoiding Time Cumulative"]
                    computing_time = finish - start
                    break

            list_makespan.append(makespan)
            list_empty_travel_time.append(empty_travel_time)
            list_avoiding_time.append(avoiding_time)
            list_computing_time.append(computing_time)

        df_makespan[prob] = list_makespan + [sum(list_makespan) / len(list_makespan)]
        df_empty_travel_time[prob] = list_empty_travel_time + [sum(list_empty_travel_time) / len(list_empty_travel_time)]
        df_avoiding_time[prob] = list_avoiding_time + [sum(list_avoiding_time) / len(list_avoiding_time)]
        df_computing_time[prob] = list_computing_time + [sum(list_computing_time) / len(list_computing_time)]
        progress += 1
        print("%.2f%% test for %s done" % (progress / len(columns) * 100, algorithm))

        writer = pd.ExcelWriter(log_dir + 'results_%s.xlsx' % algorithm)
        df_makespan.to_excel(writer, sheet_name="makespan")
        df_empty_travel_time.to_excel(writer, sheet_name="empty_travel_time")
        df_avoiding_time.to_excel(writer, sheet_name="avoiding_time")
        df_computing_time.to_excel(writer, sheet_name="computing_time")
        writer.save()
        print("test results for %s saved" % algorithm)