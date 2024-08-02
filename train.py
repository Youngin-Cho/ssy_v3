import os
import json
import string

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import deque

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
            test_env = SteelStockYard(val_dir + path, look_ahead=look_ahead,
                                      max_x=max_x, max_y=max_y, row_range=row_range, bay_range=bay_range,
                                      input_points=input_points, output_points=output_points,
                                      working_crane_ids=working_crane_ids, safety_margin=safety_margin,
                                      multi_num=multi_num, multi_w=multi_w, multi_dis=multi_dis,
                                      parameter_sharing=parameter_sharing, algorithm="RL", record_events=False, device=device)

            state, mask, crane_id = test_env.reset()
            done = False

            while not done:
                action, _, _ = agent.get_action(state, mask, crane_id)
                next_state, reward, done, next_mask, next_crane_id = test_env.step(action)

                state = next_state
                mask = next_mask
                crane_id = next_crane_id

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
        vessl.init(organization="snu-eng-dgx", project="ssy", hp=cfg)

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

    parameter_sharing = bool(cfg.parameter_sharing)
    team_reward = bool(cfg.team_reward)

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
                         parameter_sharing=parameter_sharing, algorithm="RL", record_events=record_events, device=device)

    agent = Agent(env.meta_data, env.state_size, env.num_nodes, embed_dim, num_heads,
                  num_HGT_layers, num_actor_layers, num_critic_layers, lr, lr_decay, lr_step,
                  gamma, lmbda, eps_clip, K_epoch, P_coeff, V_coeff, E_coeff, parameter_sharing, device=device)
    if cfg.vessl == 0:
        writer = SummaryWriter(log_dir)

    if cfg.load_model:
        checkpoint = torch.load(cfg.model_path)
        start_episode = checkpoint['episode'] + 1
        agent.network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    if parameter_sharing:
        with open(log_dir + "train_log.csv", 'w') as f:
            f.write('episode, reward, loss, lr\n')
    else:
        with open(log_dir + "train_log_agent1.csv", 'w') as f:
            f.write('episode, reward, loss, lr\n')
        with open(log_dir + "train_log_agent2.csv", 'w') as f:
            f.write('episode, reward, loss, lr\n')

    with open(log_dir + "validation_log.csv", 'w') as f:
        f.write('episode, makespan\n')

    for e in range(start_episode, n_episode + 1):
        if cfg.vessl == 1:
            vessl.log(payload={"Train/learnig_rate": agent.scheduler.get_last_lr()[0]}, step=e)
        elif cfg.vessl == 0:
            writer.add_scalar("Training/Learning Rate", agent.scheduler.get_last_lr()[0], e)

        done = False

        if parameter_sharing:
            r_epi = 0.0
            n = 0
            avg_loss = 0.0
        else:
            r_epi1 = 0.0
            r_epi2 = 0.0
            n1 = 0
            n2 = 0
            avg_loss1 = 0.0
            avg_loss2 = 0.0
            reward1 = []
            reward2 = []
            interval1 = []
            interval2 = []
            transition1 = []
            transition2 = []

        state, mask, crane_id = env.reset()

        while not done:
            action, action_logprob, state_value = agent.get_action(state, mask, crane_id)
            next_state, reward, done, next_mask, next_crane_id = env.step(action)

            if parameter_sharing:
                r_epi += reward
                agent.put_data((state, action, reward, next_state, action_logprob, state_value, mask, done), None)
            else:
                interval1.append(reward[2])
                interval2.append(reward[2])
                if crane_id == 0:
                    if team_reward:
                        reward1.append(reward[0] + reward[1])
                    else:
                        reward1.append(reward[0])
                else:
                    if team_reward:
                        reward2.append(reward[0] + reward[1])
                    else:
                        reward2.append(reward[1])

                if crane_id == 0:
                    if len(transition1) == 0:
                        transition1 = [state, action, action_logprob, state_value, mask, done]
                    else:
                        transition1.insert(2, sum(reward1) / sum(interval1))
                        r_epi1 += sum(reward1) / sum(interval1)
                        transition1.insert(3, next_state)
                        agent.put_data(transition1, crane_id)
                        transition1 = []
                        reward1 = []
                        interval1 = []
                else:
                    if len(transition2) == 0:
                        transition2 = [state, action, action_logprob, state_value, mask, done]
                    else:
                        transition2.insert(2, sum(reward2) / sum(interval2))
                        r_epi2 += sum(reward2) / sum(interval2)
                        transition2.insert(3, next_state)
                        agent.put_data(transition2, crane_id)
                        transition2 = []
                        reward2 = []
                        interval2 = []

            state = next_state
            mask = next_mask
            crane_id = next_crane_id

            if parameter_sharing:
                if len(agent.data) == T_horizon:
                    n += 1
                    avg_loss += agent.train(None)
            else:
                if len(agent.data1) == T_horizon:
                    n1 += 1
                    avg_loss1 += agent.train(0)
                if len(agent.data2) == T_horizon:
                    n2 += 1
                    avg_loss2 += agent.train(1)

            if done:
                break

        if parameter_sharing:
            if len(agent.data) > 0:
                n += 1
                avg_loss += agent.train(None)
        else:
            if len(agent.data1) > 0:
                n1 += 1
                avg_loss1 += agent.train(0)
            if len(agent.data2) > 0:
                n2 += 1
                avg_loss2 += agent.train(1)

        agent.scheduler.step()

        if parameter_sharing:
            print("episode: %d | reward: %.4f | loss: %.4f" % (e, r_epi, avg_loss / n))
            with open(log_dir + "train_log.csv", 'a') as f:
                f.write('%d, %1.4f, %1.4f, %f\n' % (e, r_epi, avg_loss, agent.scheduler.get_last_lr()[0]))

            if cfg.vessl == 1:
                vessl.log(payload={"Reward": r_epi, "Loss": avg_loss / n}, step=e)
            elif cfg.vessl == 0:
                writer.add_scalar("Training/Reward", r_epi, e)
                writer.add_scalar("Training/Loss", avg_loss / n, e)
        else:
            print("Agent 1) episode: %d | reward: %.4f | loss: %.4f" % (e, r_epi1, avg_loss1 / n1))
            print("Agent 2) episode: %d | reward: %.4f | loss: %.4f" % (e, r_epi2, avg_loss2 / n2))
            with open(log_dir + "train_log_agent1.csv", 'a') as f:
                f.write('%d, %1.4f, %1.4f, %f\n' % (e, r_epi1, avg_loss1 / n1, agent.scheduler.get_last_lr()[0]))
            with open(log_dir + "train_log_agent2.csv", 'a') as f:
                f.write('%d, %1.4f, %1.4f, %f\n' % (e, r_epi2, avg_loss2 / n2, agent.scheduler.get_last_lr()[0]))

            if cfg.vessl == 1:
                vessl.log(payload={"Reward1": r_epi1, "Loss1": avg_loss1 / n1}, step=e)
                vessl.log(payload={"Reward2": r_epi2, "Loss2": avg_loss2 / n2}, step=e)
            elif cfg.vessl == 0:
                writer.add_scalar("Training/Reward1", r_epi1, e)
                writer.add_scalar("Training/Reward2", r_epi2, e)
                writer.add_scalar("Training/Loss1", avg_loss1 / n1, e)
                writer.add_scalar("Training/Loss2", avg_loss2 / n2, e)


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
                                 parameter_sharing=parameter_sharing, algorithm="RL", record_events=record_events, device=device)

    if cfg.vessl == 0:
        writer.close()

