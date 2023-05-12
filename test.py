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
    algorithm = "MA" #["RL", "SD", "LD", "Random"]
    iteration = 10

    test_dirs = ["./input/test/6-10-20-4/",
                 "./input/test/8-10-20-4/",
                 "./input/test/10-10-20-4/",
                 "./input/test/12-10-20-4/",
                 "./input/test/14-10-20-4/",
                 "./input/test/10-6-20-4/",
                 "./input/test/10-8-20-4/",
                 "./input/test/10-12-20-4/",
                 "./input/test/10-14-20-4/",
                 "./input/test/10-10-16-4/",
                 "./input/test/10-10-18-4/",
                 "./input/test/10-10-22-4/",
                 "./input/test/10-10-24-4/"]

    # simulation_dir = './output/test/simulation/'
    # if not os.path.exists(simulation_dir):
    #     os.makedirs(simulation_dir)

    log_dir = './output/test/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    index = [i.split("/")[-2] for i in test_dirs]
    columns = ["P1", "P2", "P3", "P4", "P5"]
    df_makespan = pd.DataFrame(index=index, columns=columns)
    df_empty_travel_time = pd.DataFrame(index=index, columns=columns)
    df_avoiding_time = pd.DataFrame(index=index, columns=columns)

    for dir in test_dirs:
        test_paths = os.listdir(dir)
        avg_makespan = []
        avg_empty_travel_time = []
        avg_avoiding_time = []
        for j, path in enumerate(test_paths):
            makespan = 0.0
            empty_travel_time = 0.0
            avoiding_time = 0.0
            for j in range(iteration):
                df_storage = pd.read_excel(dir + path, sheet_name="storage", engine="openpyxl")
                df_reshuffle = pd.read_excel(dir + path, sheet_name="reshuffle", engine="openpyxl")
                df_retrieval = pd.read_excel(dir + path, sheet_name="retrieval", engine="openpyxl")

                env = SteelStockYard(look_ahead=2, df_storage=df_storage,
                                     df_reshuffle=df_reshuffle, df_retrieval=df_retrieval)

                if algorithm == "RL":
                    model_path = './output/train/model/episode-10000.pt'
                    device = torch.device("cpu")
                    agent = Agent(env.state_size, env.action_size, env.meta_data, n_units=128)
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

                state, info = env.reset()
                crane_in_decision = info["crane_id"]
                done = False

                while not done:
                    possible_actions = env.get_possible_actions()
                    if algorithm == "RL":
                        action = agent.get_action([state], [possible_actions], eps=0.0, noisy=False, crane_id=env.crane_in_decision)[0]
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
                        # log = env.get_logs(simulation_dir + 'event_log_{0}_{1}.csv'.format(algorithm, i))
                        log = env.get_logs()
                        makespan += log["Time"].max()
                        for crane_name in env.model.reward_info.keys():
                            empty_travel_time += env.model.reward_info[crane_name]["Empty Travel Time Cumulative"]
                            avoiding_time += env.model.reward_info[crane_name]["Avoiding Time Cumulative"]
                        break
            avg_makespan.append(makespan / iteration)
            avg_empty_travel_time.append(empty_travel_time / iteration)
            avg_avoiding_time.append(avoiding_time / iteration)

        df_makespan.loc[dir.split("/")[-2]] = avg_makespan
        df_empty_travel_time.loc[dir.split("/")[-2]] = avg_empty_travel_time
        df_avoiding_time.loc[dir.split("/")[-2]] = avg_avoiding_time
        print("done-{0}".format(dir))

    writer = pd.ExcelWriter(log_dir + 'results_{0}.xlsx'.format(algorithm))
    df_makespan.to_excel(writer, sheet_name="makespan")
    df_empty_travel_time.to_excel(writer, sheet_name="empty_travel_time")
    df_avoiding_time.to_excel(writer, sheet_name="avoiding_time")
    writer.save()