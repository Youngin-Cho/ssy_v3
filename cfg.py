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

    parser.add_argument("--n_frames", type=int, default=10000000, help="Number of frames to train")
    parser.add_argument("--eval_every", type=int, default=50000, help="Evaluate every x frames")
    parser.add_argument("--save_every", type=int, default=50000, help="Save a model every x frames")
    parser.add_argument("--n_step", type=int, default=1, help="Multistep IQN")
    parser.add_argument("--capacity", type=int, default=int(1e5), help="Replay memory size")
    parser.add_argument("--update_start", type=int, default=1000, help="Update start frame")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for updating the DQN")
    parser.add_argument("--N", type=int, default=8, help="Number of Quantiles")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--tau", type=float, default=1e-3, help="Soft update parameter tau")
    parser.add_argument("--eps_steps", type=int, default=7000, help="Linear annealed frames for Epsilon")
    parser.add_argument("--max_eps", type=float, default=1.0, help="Initial epsilon greedy value")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Final epsilon greedy value")
    parser.add_argument("--worker", type=int, default=1, help="Number of parallel Environments. Batch size increases proportional to number of worker. not recommended to have more than 4 worker")

    args = parser.parse_args()

    return args