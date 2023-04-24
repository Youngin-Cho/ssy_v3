import os
import torch
import random
import pandas as pd

from agent.iqn import *
from benchmark.heuristics import *
from environment.data import *
from environment.env import *


if __name__ == "__main__":
    algorithm = "LD" #["RL", "SD", "LD", "Random"]

    test_dir = "./input/validation/10-10-20-4/"
    test_path = os.listdir(test_dir)

    simulation_dir = './output/test/simulation/'
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    if algorithm == "RL":
        pass
        # model_path = './output/train/model/episode20000.pt'
        # device = torch.device("cpu")
        # agent = Agent()
        # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # agent.load_state_dict(checkpoint['model_state_dict'])
    elif algorithm == "SD":
        agent = shortest_distance
    elif algorithm == "LD":
        agent = longest_distance
    else:
        agent = random_selection

    makespans = []
    for i, path in enumerate(test_path):
        df_storage = pd.read_excel(test_dir + path, sheet_name="storage", engine="openpyxl")
        df_reshuffle = pd.read_excel(test_dir + path, sheet_name="reshuffle", engine="openpyxl")
        df_retrieval = pd.read_excel(test_dir + path, sheet_name="retrieval", engine="openpyxl")
        env = SteelStockYard(look_ahead=3, df_storage=df_storage,
                             df_reshuffle=df_reshuffle, df_retrieval=df_retrieval)

        state, info = env.reset()
        crane_in_decision = info["crane_id"]
        done = False

        while not done:
            possible_actions = env.get_possible_actions()
            if algorithm == "RL":
                action = agent.get_action([state], [possible_actions], eps=0.0, noisy=False, crane_id=env.crane_in_decision)
            elif algorithm == "SD":
                action = agent(state, possible_actions, crane_id=crane_in_decision)
            elif algorithm == "LD":
                action = agent(state, possible_actions, crane_id=crane_in_decision)
            else:
                action = agent(possible_actions)
            next_state, reward, done, info = env.step(action)
            state = next_state
            crane_in_decision = info["crane_id"]

            if done:
                log = env.get_logs(simulation_dir + 'event_log_{0}_{1}.csv'.format(algorithm, i))
                makespan = log["Time"].max() / len(log["Event"] == "Pick_up")
                makespans.append(makespan)
                break

    makespan_avg = sum(makespans) / len(makespans)
    print(makespan_avg)