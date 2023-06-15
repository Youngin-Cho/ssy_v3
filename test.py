import os
import torch
import random
import numpy as np
import pandas as pd

from agent.iqn import *
from benchmark.heuristics import *
from environment.data import *
from environment.env_v2 import *


if __name__ == "__main__":
    algorithm = ["RL", "SD", "MA", "Random"]
    iteration = 10

    test_dir = "./input/case_study/case1/case1-4/"
    test_paths = os.listdir(test_dir)

    # simulation_dir = './output/test/simulation/'
    # if not os.path.exists(simulation_dir):
    #     os.makedirs(simulation_dir)

    log_dir = './output/case_study/case1/case1-4/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    index = [str(i) for i in range(1, iteration + 1)] + ["avg"]
    columns = ["P%d" % i for i in range(1, len(test_paths) + 1)]

    for name in algorithm:
        random.seed(42)
        progress = 0
        df_makespan = pd.DataFrame(index=index, columns=columns)
        df_empty_travel_time = pd.DataFrame(index=index, columns=columns)
        df_avoiding_time = pd.DataFrame(index=index, columns=columns)

        for prob, path in zip(columns, test_paths):
            list_makespan = []
            list_empty_travel_time = []
            list_avoiding_time = []

            for j in range(iteration):
                makespan = 0.0
                empty_travel_time = 0.0
                avoiding_time = 0.0

                df_storage = pd.read_excel(test_dir + path, sheet_name="storage", engine="openpyxl")
                df_reshuffle = pd.read_excel(test_dir + path, sheet_name="reshuffle", engine="openpyxl")
                df_retrieval = pd.read_excel(test_dir + path, sheet_name="retrieval", engine="openpyxl")

                env = SteelStockYard(look_ahead=2, df_storage=df_storage,
                                     df_reshuffle=df_reshuffle, df_retrieval=df_retrieval)

                if name == "RL":
                    model_path = './output/train/model/episode-10000.pt'
                    device = torch.device("cpu")
                    agent = Agent(env.state_size, env.action_size, env.meta_data, n_units=128)
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
                elif name == "SD":
                    agent = shortest_distance
                elif name == "LD":
                    agent = longest_distance
                elif name == "MA":
                    agent = minimize_avoiding_time
                else:
                    agent = random_selection

                state, info = env.reset()
                crane_in_decision = info["crane_id"]
                done = False

                while not done:
                    possible_actions = env.get_possible_actions()
                    if name == "RL":
                        action = agent.get_action([state], [possible_actions], eps=0.0,
                                                  noisy=False, crane_id=env.crane_in_decision)[0]
                    elif name == "SD":
                        action = agent(state, possible_actions, crane_id=crane_in_decision)
                    elif name == "LD":
                        action = agent(state, possible_actions, crane_id=crane_in_decision)
                    elif name == "MA":
                        action = agent(state, possible_actions, crane_id=crane_in_decision)
                    else:
                        action = agent(possible_actions)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    crane_in_decision = info["crane_id"]

                    if done:
                        # log = env.get_logs(simulation_dir + 'event_log_{0}_{1}.csv'.format(algorithm, i))
                        log = env.get_logs()
                        makespan = log["Time"].max()
                        for crane_name in env.model.reward_info.keys():
                            empty_travel_time += env.model.reward_info[crane_name]["Empty Travel Time Cumulative"]
                            avoiding_time += env.model.reward_info[crane_name]["Avoiding Time Cumulative"]
                        break

                list_makespan.append(makespan)
                list_empty_travel_time.append(empty_travel_time)
                list_avoiding_time.append(avoiding_time)

            df_makespan[prob] = list_makespan + [sum(list_makespan) / len(list_makespan)]
            df_empty_travel_time[prob] = list_empty_travel_time + [sum(list_empty_travel_time) / len(list_empty_travel_time)]
            df_avoiding_time[prob] = list_avoiding_time + [sum(list_avoiding_time) / len(list_avoiding_time)]
            progress += 1
            print("%.2f%% test for %s done" % (progress / len(columns) * 100, name))

        writer = pd.ExcelWriter(log_dir + 'results_%s.xlsx' % name)
        df_makespan.to_excel(writer, sheet_name="makespan")
        df_empty_travel_time.to_excel(writer, sheet_name="empty_travel_time")
        df_avoiding_time.to_excel(writer, sheet_name="avoiding_time")
        writer.save()
        print("test results for %s saved" % name)