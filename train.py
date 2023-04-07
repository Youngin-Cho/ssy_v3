import os
import vessl
import pandas as pd

from cfg import get_cfg
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

        state = test_env.reset()
        done = False

        while not done:
            possible_actions = test_env.get_possible_actions()
            action = agent.get_action([state], [possible_actions], eps=0.0)
            next_state, r, done = test_env.step(action[0])
            state = next_state

            if done:
                log = test_env.get_logs()
                makespan = log["Time"].max() / len(log["Event"] == "Pick_up")
                makespans.append(makespan)
                break

    return np.mean(makespans)


if __name__ == "__main__":
    cfg = get_cfg()
    vessl.init(organization="snu-eng-dgx", project="S", hp=cfg)

    n_frames = cfg.n_frames
    eval_every = cfg.eval_every
    save_every = cfg.save_every

    look_ahead = cfg.look_ahead
    n_stor_to = cfg.n_stor_to
    n_resh_from = cfg.n_resh_from
    n_resh_to = cfg.n_resh_to
    n_retr_from = cfg.n_retr_from

    n_step = cfg.n_step
    buffer_size = cfg.buffer_size
    batch_size = cfg.batch_size
    N = cfg.N
    lr = cfg.lr
    gamma = cfg.gamma
    tau = cfg.tau
    eps_steps = cfg.eps_steps
    max_eps = cfg.max_eps
    min_eps = cfg.min_eps
    worker = cfg.worker

    model_dir = '/output/train/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = '/output/train/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    validation_dir = './input/validation/{0}-{1}-{2}-{3}/'.format(n_stor_to, n_resh_from, n_resh_to, n_retr_from)

    envs = ParallelEnv(worker, look_ahead=look_ahead,
                       num_of_storage_to_piles=n_stor_to, num_of_reshuffle_from_piles=n_resh_from,
                       num_of_reshuffle_to_piles=n_resh_to, num_of_retrieval_from_piles=n_retr_from)

    agent = Agent(envs.state_size, envs.action_size, envs.meta_data, look_ahead,
                  n_step, batch_size, buffer_size, lr, tau, gamma, N, worker)
    # writer = SummaryWriter(log_dir)

    if cfg.load_model:
        checkpoint = torch.load(cfg.model_path)
        start_frame = checkpoint['frame'] + 1
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_frame = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss\n')

    eps = min_eps
    d_eps = max_eps - min_eps

    state = envs.reset()
    episode = 1
    reward_list = []
    loss_list = []

    for frame in range(start_frame, n_frames + 1):
        possible_actions = envs.get_possible_actions()
        action = agent.get_action(state, possible_actions, eps=eps)
        next_state, reward, done = envs.step(action)  # returns np.stack(obs), np.stack(action) ...
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            loss = agent.step(s, a, r, ns, d)
        state = next_state

        reward_list.append(np.mean(reward))
        if loss is not None:
            loss_list.append(loss)

        eps = max(max_eps - ((frame * d_eps) / eps_steps), min_eps)

        if frame % eval_every == 0 or frame == 1:
            makespan = evaluate(validation_dir)
            vessl.log(payload={"Perf/makespan": makespan}, step=episode)

        if frame % save_every == 0:
            agent.save(frame, model_dir)

        if True in done:
            reward_epi = np.sum(reward_list)
            avg_loss = np.mean(loss_list)

            print("episode: %d | reward: %.2f | loss : %.2f" % (episode, reward_epi, avg_loss))
            with open(log_dir + "train_log.csv", 'a') as f:
                f.write('%d,%1.2f,1.2%f\n' % (episode, reward_epi, avg_loss))

            vessl.log(payload={"Train/reward": reward_epi, "Train/loss": avg_loss}, step=episode)

            reward_list = []
            loss_list = []
            episode += 1
            state = envs.reset()
