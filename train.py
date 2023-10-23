import os
import json
import vessl
import torch
import string
import numpy as np
import pandas as pd

from cfg_train import get_cfg
# from torch.utils.tensorboard import SummaryWriter
from agent.iqn import Agent
from environment.data import DataGenerator
from environment.env import SteelStockYard


def evaluate(validation_dir):
    validation_path = os.listdir(validation_dir)
    makespans = []

    for path in validation_path:
        data_src = validation_dir + path
        test_env = SteelStockYard(data_src, look_ahead=look_ahead,
                                  max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                  input_points=input_points, output_points=output_points,
                                  working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                                  multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis, record_events=False)

        state, info = test_env.reset()
        crane_in_decision = info["crane_id"]
        done = False

        while not done:
            possible_actions = test_env.get_possible_actions()
            action = agent.get_action([state], [possible_actions], eps=0.0, noisy=False, crane_id=crane_in_decision)
            next_state, r, done, info = test_env.step(action[0])
            state = next_state
            crane_in_decision = info["crane_id"]

            if done:
                makespan = test_env.model.env.now / test_env.model.num_plates_cum
                # log = test_env.get_logs()
                # makespan = log["Time"].max() / len(log["Event"] == "Pick_up")
                makespans.append(makespan)
                break

    return sum(makespans) / len(makespans)


if __name__ == "__main__":
    cfg = get_cfg()
    vessl.init(organization="snu-eng-dgx", project="ssy", hp=cfg)

    look_ahead = cfg.look_ahead
    record_events = cfg.record_events

    n_rows = cfg.n_rows
    storage = cfg.storage
    reshuffle = cfg.reshuffle
    retrieval = cfg.retrieval
    n_bays_in_area1 = cfg.n_bays_in_area1
    n_bays_in_area2 = cfg.n_bays_in_area2
    n_bays_in_area3 = cfg.n_bays_in_area3
    n_bays_in_area4 = cfg.n_bays_in_area4
    n_bays_in_area5 = cfg.n_bays_in_area5
    n_bays_in_area6 = cfg.n_bays_in_area6
    n_from_piles_storage = cfg.n_from_piles_storage
    n_to_piles_storage = cfg.n_to_piles_storage
    n_from_piles_reshuffle = cfg.n_from_piles_reshuffle
    n_to_piles_reshuffle = cfg.n_to_piles_reshuffle
    n_from_piles_retrieval_cn1 = cfg.n_from_piles_retrieval_cn1
    n_from_piles_retrieval_cn2 = cfg.n_from_piles_retrieval_cn2
    n_from_piles_retrieval_cn3 = cfg.n_from_piles_retrieval_cn3
    n_plates_storage = cfg.n_plates_storage
    n_plates_reshuffle = cfg.n_plates_reshuffle
    n_plates_retrieval = cfg.n_plates_retrieval

    max_x = n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6 + 4
    max_y = cfg.n_rows
    row_range = (string.ascii_uppercase[0], string.ascii_uppercase[cfg.n_rows - 1])
    bay_range = (1, n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6)
    input_points = (1,)
    output_points = (1 + n_bays_in_area1 + n_bays_in_area2 + 1,
                     1 + n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + 2,
                     1 + n_bays_in_area1 + n_bays_in_area2 + n_bays_in_area3 + n_bays_in_area4 + n_bays_in_area5 + n_bays_in_area6 + 3)
    working_crane_ids = tuple()
    if cfg.is_crane1_working:
        working_crane_ids = working_crane_ids + ("Crane-1", )
    if cfg.is_crane2_working:
        working_crane_ids = working_crane_ids + ("Crane-2", )
    safety_margin = cfg.safety_margin

    multi_num = cfg.multi_num
    multi_w = cfg.multi_w
    multi_dis = cfg.multi_dis

    n_episode = cfg.n_episode
    eval_every = cfg.eval_every
    save_every = cfg.save_every
    new_instance_every = cfg.new_instance_every

    n_units = cfg.n_units
    n_step = cfg.n_step
    capacity = cfg.capacity
    alpha = cfg.alpha
    beta_start = cfg.beta_start
    beta_steps = cfg.beta_steps
    batch_size = cfg.batch_size
    N = cfg.N
    lr = cfg.lr
    lr_step = cfg.lr_step
    lr_decay = cfg.lr_decay
    gamma = cfg.gamma
    tau = cfg.tau

    model_dir = '/output/train/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = '/output/train/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    validation_dir = './input/data/validation_without_retrieval/'

    with open(log_dir + "parameters.json", 'w') as f:
        json.dump(vars(cfg), f, indent=4)


    data_src = DataGenerator(rows=tuple(i for i in string.ascii_uppercase[:cfg.n_rows]),
                             storage=storage,
                             reshuffle=reshuffle,
                             retrieval=retrieval,
                             n_bays_in_area1=n_bays_in_area1,
                             n_bays_in_area2=n_bays_in_area2,
                             n_bays_in_area3=n_bays_in_area3,
                             n_bays_in_area4=n_bays_in_area4,
                             n_bays_in_area5=n_bays_in_area5,
                             n_bays_in_area6=n_bays_in_area6,
                             n_from_piles_storage=n_from_piles_storage,
                             n_to_piles_storage=n_to_piles_storage,
                             n_from_piles_reshuffle=n_from_piles_reshuffle,
                             n_to_piles_reshuffle=n_to_piles_reshuffle,
                             n_from_piles_retrieval_cn1=n_from_piles_retrieval_cn1,
                             n_from_piles_retrieval_cn2=n_from_piles_retrieval_cn2,
                             n_from_piles_retrieval_cn3=n_from_piles_retrieval_cn3,
                             n_plates_storage=n_plates_storage,
                             n_plates_reshuffle=n_plates_reshuffle,
                             n_plates_retrieval=n_plates_retrieval,
                             working_crane_ids=working_crane_ids,
                             safety_margin=safety_margin)
    env = SteelStockYard(data_src, look_ahead=look_ahead,
                         max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                         input_points=input_points, output_points=output_points,
                         working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                         multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis, record_events=record_events)

    agent = Agent(env.state_size, env.action_size, env.meta_data, look_ahead, n_units,
                  capacity, alpha, beta_start, beta_steps,
                  n_step, batch_size, lr, lr_step, lr_decay, tau, gamma, N)
    # writer = SummaryWriter(log_dir)

    if cfg.load_model:
        checkpoint = torch.load(cfg.model_path)
        start_episode = checkpoint['episode'] + 1
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss\n')
    with open(log_dir + "validation_log.csv", 'w') as f:
        f.write('frame, makespan\n')

    for episode in range(start_episode, n_episode + 1):
        reward_tot = 0.0
        done = False
        loss_list = []

        state, info = env.reset()
        crane_in_decision = info["crane_id"]

        while True:
            possible_actions = env.get_possible_actions()
            action = agent.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=crane_in_decision)
            next_state, reward, done, info = env.step(action[0])
            loss = agent.step(state, action[0], reward, next_state, done)
            if loss is not None:
                loss_list.append(loss)

            state = next_state
            reward_tot += reward
            crane_in_decision = info["crane_id"]

            if done:

                loss_avg = sum(loss_list) / len(loss_list)
                print("episode: %d | total_rewards: %.2f | loss: %.2f" % (episode, reward_tot, loss_avg))

                vessl.log(payload={"LearnigRate": agent.scheduler.get_last_lr()[0]}, step=episode)
                vessl.log(payload={"Reward": reward_tot}, step=episode)
                vessl.log(payload={"Loss": loss_avg}, step=episode)
                # writer.add_scalar("Training/Epsilon", eps, episode)
                # writer.add_scalar("Training/Reward", reward_tot, episode)
                # writer.add_scalar("Training/Loss", loss_avg, episode)

                break

        if episode % eval_every == 0 or episode == 1:
            makespan = evaluate(validation_dir)
            vessl.log(payload={"Makespan": makespan}, step=episode)
            # writer.add_scalar("Validation/Makespan", makespan, episode)
            with open(log_dir + "validation_log.csv", 'a') as f:
                f.write('%d,%1.2f\n' % (episode, makespan))

        if episode % save_every == 0:
            agent.save(episode, model_dir)

        if episode % new_instance_every == 0:
            env = SteelStockYard(data_src, look_ahead=look_ahead,
                                 max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                 input_points=input_points, output_points=output_points,
                                 working_crane_ids=working_crane_ids, safety_margin=safety_margin)

        agent.scheduler.step()

    # writer.close()