import os
import vessl
import numpy as np
import pandas as pd

from cfg_single import get_cfg
from torch.utils.tensorboard import SummaryWriter
from agent.iqn import *
from environment.env import *
from environment.multi_process import *


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
            action = agent.get_action([state], [possible_actions], eps=0.0, noisy=False, crane_id=crane_in_decision)
            next_state, r, done, info = test_env.step(action[0])
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
    # base_lr = cfg.base_lr
    # max_lr = cfg.max_lr
    # step_size_up = cfg.step_size_up
    # step_size_down = cfg.step_size_down
    gamma = cfg.gamma
    tau = cfg.tau
    worker = cfg.worker

    model_dir = '/output/train/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = '/output/train/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    validation_dir = './input/validation/{0}-{1}-{2}-{3}/'.format(n_stor_to, n_resh_from, n_resh_to, n_retr_from)

    env = SteelStockYard(look_ahead=look_ahead,
                         num_of_storage_to_piles=n_stor_to, num_of_reshuffle_from_piles=n_resh_from,
                         num_of_reshuffle_to_piles=n_resh_to, num_of_retrieval_from_piles=n_retr_from)

    agent = Agent(env.state_size, env.action_size, env.meta_data, look_ahead,
                  capacity, alpha, beta_start, beta_steps,
                  n_step, batch_size, lr, lr_step, lr_decay, tau, gamma, N, worker)
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

        # sample_crane1 = []
        # sample_crane2 = []

        loss_list = []
        state, info = env.reset()
        crane_in_decision = info["crane_id"]

        while True:
            possible_actions = env.get_possible_actions()
            action = agent.get_action([state], [possible_actions], eps=0.0, noisy=True, crane_id=crane_in_decision)
            next_state, reward, done, info = env.step(action[0])

            # if info["crane_id"] == 0:
            #     if len(sample_crane1) == 0:
            #         sample_crane1.append(state)
            #         loss = None
            #     else:
            #         sample_crane1 = sample_crane1 + [action[0], reward, next_state, done, crane_in_decision]
            #         loss = agent.step(*sample_crane1)
            #         sample_crane1 = [next_state]
            # else:
            #     if len(sample_crane2) == 0:
            #         sample_crane2.append(state)
            #         loss = None
            #     else:
            #         sample_crane2 = sample_crane2 + [action[0], reward, next_state, done, crane_in_decision]
            #         loss = agent.step(*sample_crane2)
            #         sample_crane2 = [next_state]
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

        agent.scheduler.step()

    # writer.close()