import os
import json
import string

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from cfg_train import get_cfg
from agent.ppo import *
from environment.data import DataGenerator
from environment.env import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate(val_dir):
    val_paths = os.listdir(val_dir)
    with torch.no_grad():
        makespan_lst = []
        for path in val_paths:
            test_env = SteelStockYard(data_src, look_ahead=look_ahead,
                                      max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                      input_points=input_points, output_points=output_points,
                                      working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                                      multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                                      reward_sig=reward_sig, rl=True, record_events=False, device=device)

            state, mask = test_env.reset()
            done = False

            while not done:
                action, _, _ = agent.get_action(state, mask)
                next_state, reward, done, next_mask = test_env.step(action)

                state = next_state
                mask = next_mask

                if done:
                    # log = test_env.get_logs()
                    break

            makespan = test_env.model.env.now / test_env.model.num_plates_cum
            makespan_lst.append(makespan)

        average_makespan = sum(makespan_lst) / len(makespan_lst)

        return average_makespan


if __name__ == "__main__":
    date = datetime.now().strftime('%m%d_%H_%M')
    cfg = get_cfg()
    if cfg.vessl == 1:
        import vessl
        vessl.init(organization="snu-eng-dgx", project="quay", hp=cfg)

    look_ahead = cfg.look_ahead
    record_events = bool(cfg.record_events)

    n_rows = cfg.n_rows
    storage = bool(cfg.storage)
    reshuffle = bool(cfg.reshuffle)
    retrieval = bool(cfg.retrieval)
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
        working_crane_ids = working_crane_ids + ("Crane-1",)
    if cfg.is_crane2_working:
        working_crane_ids = working_crane_ids + ("Crane-2",)
    safety_margin = cfg.safety_margin

    multi_num = cfg.multi_num
    multi_w = cfg.multi_w
    multi_dis = cfg.multi_dis

    reward_sig = cfg.reward_sig

    n_episode = cfg.n_episode
    eval_every = cfg.eval_every
    save_every = cfg.save_every
    new_instance_every = cfg.new_instance_every

    embed_dim = cfg.embed_dim
    num_heads = cfg.num_heads
    num_HGT_layers = cfg.num_HGT_layers
    num_actor_layers = cfg.num_actor_layers
    num_critic_layers = cfg.num_critic_layers
    lr = cfg.lr
    lr_decay = cfg.lr_decay
    lr_step = cfg.lr_step
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    eps_clip = cfg.eps_clip
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon
    P_coeff = cfg.P_coeff
    V_coeff = cfg.V_coeff
    E_coeff = cfg.E_coeff

    eval_every = cfg.eval_every
    save_every = cfg.save_every
    new_instance_every = cfg.new_instance_every

    val_dir = cfg.val_dir

    if cfg.vessl == 1:
        model_dir = '/output/train/' + date + '/model/'
    elif cfg.vessl == 0:
        model_dir = './output/train/' + date + '/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if cfg.vessl == 1:
        log_dir = '/output/train/' + date + '/log/'
    elif cfg.vessl == 0:
        log_dir = './output/train/' + date + '/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # simulation_dir = '../output/train/simulation/'
    # if not os.path.exists(simulation_dir):
    #    os.makedirs(simulation_dir)

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
                         multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                         reward_sig=reward_sig, rl=True, record_events=record_events, device=device)

    agent = Agent(env.meta_data, env.state_size, env.num_nodes, embed_dim, num_heads,
                  num_HGT_layers, num_actor_layers, num_critic_layers, lr, lr_decay, lr_step,
                  gamma, lmbda, eps_clip, K_epoch, P_coeff, V_coeff, E_coeff, device=device)
    if cfg.vessl == 0:
        writer = SummaryWriter(log_dir)

    if cfg.load_model:
        checkpoint = torch.load(cfg.model_path)
        start_episode = checkpoint['episode'] + 1
        agent.network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss, lr\n')
    with open(log_dir + "validation_log.csv", 'w') as f:
        f.write('episode, average_delay, move_ratio, priority_ratio\n')

    for e in range(start_episode, n_episode + 1):
        if cfg.vessl == 1:
            vessl.log(payload={"Train/learnig_rate": agent.scheduler.get_last_lr()[0]}, step=e)
        elif cfg.vessl == 0:
            writer.add_scalar("Training/Learning Rate", agent.scheduler.get_last_lr()[0], e)

        n = 0
        r_epi = 0.0
        avg_loss = 0.0
        done = False

        state, mask = env.reset()

        while not done:
            for t in range(T_horizon):
                action, action_logprob, state_value = agent.get_action(state, mask)
                next_state, reward, done, next_mask = env.step(action)

                agent.put_data((state, action, reward, next_state, action_logprob, state_value, mask, done))
                state = next_state
                mask = next_mask

                r_epi += reward

                if done:
                    break

            n += 1
            avg_loss += agent.train()
        agent.scheduler.step()

        print("episode: %d | reward: %.4f | loss: %.4f" % (e, r_epi, avg_loss / n))
        with open(log_dir + "train_log.csv", 'a') as f:
            f.write('%d, %1.4f, %1.4f, %f\n' % (e, r_epi, avg_loss, agent.scheduler.get_last_lr()[0]))

        if cfg.vessl == 1:
            vessl.log(payload={"Reward": r_epi, "Loss": avg_loss / n}, step=e)
        elif cfg.vessl == 0:
            writer.add_scalar("Training/Reward", r_epi, e)
            writer.add_scalar("Training/Loss", avg_loss / n, e)

        if e == start_episode or e % eval_every == 0:
            makespan = evaluate(val_dir)

            with open(log_dir + "validation_log.csv", 'a') as f:
                f.write('%d,%1.2f\n' % (e, makespan))

            if cfg.vessl == 1:
                vessl.log(payload={"Makespan": makespan}, step=e)
            elif cfg.vessl == 0:
                writer.add_scalar("Validation/Makespan", makespan, e)

        if e % save_every == 0:
            agent.save_network(e, model_dir)

        if e % new_instance_every == 0:
            env = SteelStockYard(data_src, look_ahead=look_ahead,
                                 max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                 input_points=input_points, output_points=output_points,
                                 working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                                 multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                                 reward_sig=reward_sig, rl=True, record_events=record_events, device=device)

    if cfg.vessl == 0:
        writer.close()

