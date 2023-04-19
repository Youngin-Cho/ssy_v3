import os
import vessl
import numpy as np
import pandas as pd

from cfg_single import get_cfg
from torch.utils.tensorboard import SummaryWriter
from agent.iqn import *
from environment.env import *
from benchmark.heuristics import shortest_distance


def evaluate(validation_dir):
    validation_path = os.listdir(validation_dir)
    makespans = []

    for path in validation_path:
        df_storage = pd.read_excel(validation_dir + path, sheet_name="storage", engine="openpyxl")
        df_reshuffle = pd.read_excel(validation_dir + path, sheet_name="reshuffle", engine="openpyxl")
        df_retrieval = pd.read_excel(validation_dir + path, sheet_name="retrieval", engine="openpyxl")
        test_env = SteelStockYard(look_ahead=look_ahead, df_storage=df_storage,
                                  df_reshuffle=df_reshuffle, df_retrieval=df_retrieval)

        state, info = test_env.reset()
        crane_in_decision = info["crane_id"]
        done = False

        while not done:
            possible_actions = test_env.get_possible_actions()

            if mode == "both":
                if crane_in_decision == 0:
                    action = agent_crane1.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=env.crane_in_decision)[0]
                else:
                    action = agent_crane2.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=env.crane_in_decision)[0]
            elif mode == "crane1":
                if crane_in_decision == 0:
                    action = agent_crane1.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=env.crane_in_decision)[0]
                else:
                    action = agent_crane2(state, possible_actions, crane_id=crane_in_decision)
            else:
                if crane_in_decision == 0:
                    action = agent_crane1(state, possible_actions, crane_id=crane_in_decision)
                else:
                    action = agent_crane2.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=env.crane_in_decision)[0]

            next_state, r, done, info = test_env.step(action)
            state = next_state
            crane_in_decision = info["crane_id"]

            if done:
                log = test_env.get_logs()
                makespan = log["Time"].max() / len(log["Event"] == "Pick_up")
                makespans.append(makespan)
                break

    return sum(makespans) / len(makespans)


if __name__ == "__main__":
    cfg = get_cfg()
    vessl.init(organization="snu-eng-dgx", project="S", hp=cfg)

    mode = cfg.mode
    n_episode = cfg.n_episode
    eval_every = cfg.eval_every
    save_every = cfg.save_every

    look_ahead = cfg.look_ahead
    n_stor_to = cfg.n_stor_to
    n_resh_from = cfg.n_resh_from
    n_resh_to = cfg.n_resh_to
    n_retr_from = cfg.n_retr_from

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
    worker = cfg.worker

    model_dir = '/output/train/model/'
    if not os.path.exists(model_dir):
        if mode == "both":
            os.makedirs(model_dir + 'crane1/')
            os.makedirs(model_dir + 'crane2/')
        elif mode == "crane1":
            os.makedirs(model_dir + 'crane1/')
        elif mode == "crane2":
            os.makedirs(model_dir + 'crane2/')
        else:
            raise Exception("Invalid Input")

    log_dir = '/output/train/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    validation_dir = './input/validation/{0}-{1}-{2}-{3}/'.format(n_stor_to, n_resh_from, n_resh_to, n_retr_from)

    env = SteelStockYard(look_ahead=look_ahead,
                         num_of_storage_to_piles=n_stor_to, num_of_reshuffle_from_piles=n_resh_from,
                         num_of_reshuffle_to_piles=n_resh_to, num_of_retrieval_from_piles=n_retr_from)

    if mode == "both":
        agent_crane1 = Agent(env.state_size, env.action_size, env.meta_data, look_ahead,
                             capacity, alpha, beta_start, beta_steps,
                             n_step, batch_size, lr, lr_step, lr_decay, tau, gamma, N, worker)
        agent_crane2 = Agent(env.state_size, env.action_size, env.meta_data, look_ahead,
                             capacity, alpha, beta_start, beta_steps,
                             n_step, batch_size, lr, lr_step, lr_decay, tau, gamma, N, worker)
    elif mode == "crane1":
        agent_crane1 = Agent(env.state_size, env.action_size, env.meta_data, look_ahead,
                             capacity, alpha, beta_start, beta_steps,
                             n_step, batch_size, lr, lr_step, lr_decay, tau, gamma, N, worker)
        agent_crane2 = shortest_distance
    else:
        agent_crane1 = shortest_distance
        agent_crane2 = Agent(env.state_size, env.action_size, env.meta_data, look_ahead,
                             capacity, alpha, beta_start, beta_steps,
                             n_step, batch_size, lr, lr_step, lr_decay, tau, gamma, N, worker)

    # writer = SummaryWriter(log_dir)

    # if cfg.load_model:
    #     checkpoint_crane1 = torch.load(cfg.model_path)
    #     start_episode = checkpoint['episode'] + 1
    #     agent_crane1.qnetwork_local.load_state_dict(checkpoint['model1_state_dict'])
    #     agent_crane1.qnetwork_target.load_state_dict(checkpoint['model1_state_dict'])
    #     agent_crane1.optimizer.load_state_dict(checkpoint['optimizer1_state_dict'])
    #     agent_crane2.qnetwork_local.load_state_dict(checkpoint['model2_state_dict'])
    #     agent_crane2.qnetwork_target.load_state_dict(checkpoint['model2_state_dict'])
    #     agent_crane2.optimizer.load_state_dict(checkpoint['optimizer2_state_dict'])
    # else:
    #     start_episode = 1
    start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss\n')
    with open(log_dir + "validation_log.csv", 'w') as f:
        f.write('frame, makespan\n')

    for episode in range(start_episode, n_episode + 1):
        reward_tot = 0.0
        done = False

        if mode == "both":
            sample_crane1 = []
            sample_crane2 = []
            loss_crane1_list = []
            loss_crane2_list = []
        elif mode == "crane1":
            sample_crane1 = []
            loss_crane1_list = []
        else:
            sample_crane2 = []
            loss_crane2_list = []

        state, info = env.reset()
        crane_in_decision = info["crane_id"]

        while True:
            possible_actions = env.get_possible_actions()
            if mode == "both":
                if crane_in_decision == 0:
                    action = agent_crane1.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=env.crane_in_decision)[0]
                else:
                    action = agent_crane2.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=env.crane_in_decision)[0]
            elif mode == "crane1":
                if crane_in_decision == 0:
                    action = agent_crane1.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=env.crane_in_decision)[0]
                else:
                    action = agent_crane2(state, possible_actions, crane_id=crane_in_decision)
            else:
                if crane_in_decision == 0:
                    action = agent_crane1(state, possible_actions, crane_id=crane_in_decision)
                else:
                    action = agent_crane2.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=env.crane_in_decision)[0]

            next_state, reward, done, info = env.step(action)

            if mode == "both":
                if info["crane_id"] == 0:
                    if len(sample_crane1) == 0:
                        sample_crane1.append(state)
                    else:
                        sample_crane1 = sample_crane1 + [action, reward, next_state, done]
                        loss = agent_crane1.step(*sample_crane1)
                        sample_crane1 = [next_state]
                        if loss is not None:
                            loss_crane1_list.append(loss)
                else:
                    if len(sample_crane2) == 0:
                        sample_crane2.append(state)
                    else:
                        sample_crane2 = sample_crane2 + [action, reward, next_state, done]
                        loss = agent_crane2.step(*sample_crane2)
                        sample_crane2 = [next_state]
                        if loss is not None:
                            loss_crane2_list.append(loss)
            elif mode == "crane1":
                if info["crane_id"] == 0:
                    if len(sample_crane1) == 0:
                        sample_crane1.append(state)
                    else:
                        sample_crane1 = sample_crane1 + [action, reward, next_state, done]
                        loss = agent_crane1.step(*sample_crane1)
                        sample_crane1 = [next_state]
                        if loss is not None:
                            loss_crane1_list.append(loss)
            else:
                if len(sample_crane2) == 0:
                    sample_crane2.append(state)
                else:
                    sample_crane2 = sample_crane2 + [action, reward, next_state, done]
                    loss = agent_crane2.step(*sample_crane2)
                    sample_crane2 = [next_state]
                    if loss is not None:
                        loss_crane2_list.append(loss)

            state = next_state
            reward_tot += reward
            crane_in_decision = info["crane_id"]

            if done:
                print("episode: %d | total_rewards: %.2f" % (episode, reward_tot))
                if mode == "both":
                    loss_crane1_avg = sum(loss_crane1_list) / len(loss_crane1_list)
                    loss_crane2_avg = sum(loss_crane2_list) / len(loss_crane2_list)
                elif mode == "crane1":
                    loss_crane1_avg = sum(loss_crane1_list) / len(loss_crane1_list)
                else:
                    loss_crane2_avg = sum(loss_crane2_list) / len(loss_crane2_list)

                vessl.log(payload={"Reward": reward_tot}, step=episode)
                if mode == "both":
                    vessl.log(payload={"LearnigRate": agent_crane1.scheduler.get_last_lr()[0]}, step=episode)
                    vessl.log(payload={"Loss/Crane1": loss_crane1_avg}, step=episode)
                    vessl.log(payload={"Loss/Crane2": loss_crane2_avg}, step=episode)
                elif mode == "crane1":
                    vessl.log(payload={"LearnigRate": agent_crane1.scheduler.get_last_lr()[0]}, step=episode)
                    vessl.log(payload={"Loss/Crane1": loss_crane1_avg}, step=episode)
                else:
                    vessl.log(payload={"LearnigRate": agent_crane2.scheduler.get_last_lr()[0]}, step=episode)
                    vessl.log(payload={"Loss/Crane2": loss_crane2_avg}, step=episode)
                # writer.add_scalar("Training/Epsilon", eps, episode)
                # writer.add_scalar("Training/Reward", reward_tot, episode)
                # writer.add_scalar("Training/Loss", loss_avg, episode)

                break

        # eps = max(max_eps - ((episode * d_eps) / eps_steps), min_eps)

        if episode % eval_every == 0 or episode == 1:
            makespan = evaluate(validation_dir)
            vessl.log(payload={"Makespan": makespan}, step=episode)
            # writer.add_scalar("Validation/Makespan", makespan, episode)
            with open(log_dir + "validation_log.csv", 'a') as f:
                f.write('%d,%1.2f\n' % (episode, makespan))

        if episode % save_every == 0:
            if mode == "both":
                agent_crane1.save(episode, model_dir + "crane1/")
                agent_crane2.save(episode, model_dir + "crane2/")
            elif mode == "crane1":
                agent_crane1.save(episode, model_dir + "crane1/")
            else:
                agent_crane2.save(episode, model_dir + "crane2/")

        if mode == "both":
            agent_crane1.scheduler.step()
            agent_crane2.scheduler.step()
        elif mode == "crane1":
            agent_crane1.scheduler.step()
        else:
            agent_crane2.scheduler.step()

    # writer.close()