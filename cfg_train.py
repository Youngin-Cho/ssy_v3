import argparse


def get_cfg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--look_ahead", type=int, default=2, help="number of steel plates included in states")
    parser.add_argument("--n_stor_to", type=int, default=10, help="number of storage to piles")
    parser.add_argument("--n_resh_from", type=int, default=10, help="number of reshuffle from piles")
    parser.add_argument("--n_resh_to", type=int, default=20, help="number of reshuffle to piles")
    parser.add_argument("--n_retr_from", type=int, default=4, help="number of retreival from piles")

    parser.add_argument("--mode", type=str, default="both", help="training mode")
    parser.add_argument("--n_episode", type=int, default=10000, help="Number of episodes to train")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every x frames")
    parser.add_argument("--save_every", type=int, default=1000, help="Save a model every x frames")
    parser.add_argument("--n_units", type=int, default=256, help="Number of units in hidden layers of IQN")
    parser.add_argument("--n_step", type=int, default=10, help="Multistep IQN")
    parser.add_argument("--capacity", type=int, default=1000, help="Replay memory size")
    parser.add_argument("--alpha", type=float, default=0.6, help="Control paramter for priorizted sampling")
    parser.add_argument("--beta_start", type=float, default=0.4, help="Correction parameter for importance sampling")
    parser.add_argument("--beta_steps", type=int, default=100000, help="Total number of steps for annealing")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updating the DQN")
    parser.add_argument("--N", type=int, default=8, help="Number of Quantiles")
    # parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    # parser.add_argument("--lr_step", type=int, default=2500, help="Step size to reduce learning rate")
    # parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate decay ratio")
    parser.add_argument("--base_lr", type=float, default=0.00001, help="Base learning rate")
    parser.add_argument("--max_lr", type=float, default=0.000001, help="Maximum learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor gamma")
    parser.add_argument("--step_size_up", type=int, default=300, help="Number of episodes increasing a learning rate")
    parser.add_argument("--step_size_down", type=int, default=700, help="Number of episodes decreasing a learning rate")
    parser.add_argument("--tau", type=float, default=0.001, help="Soft update parameter tau")
    parser.add_argument("--eps_steps", type=int, default=20000, help="Linear annealed frames for Epsilon")
    parser.add_argument("--max_eps", type=float, default=1.0, help="Initial epsilon greedy value")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Final epsilon greedy value")
    parser.add_argument("--worker", type=int, default=1, help="Number of parallel Environments. Batch size increases proportional to number of worker. not recommended to have more than 4 worker")

    args = parser.parse_args()

    return args