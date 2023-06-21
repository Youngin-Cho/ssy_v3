import argparse


def get_cfg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="model file path")
    parser.add_argument("--data_folder", type=str, default=None, help="test data path")

    parser.add_argument("--n_units", type=int, default=256, help="number of units in hidden layers of IQN")
    parser.add_argument("--algorithm", type=str, default="RL", help="test algorithm")
    parser.add_argument("--iteration", type=int, default=10, help="number of iterations")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    parser.add_argument("--safe_margin", type=int, default=5, help="safety margin between two cranes")

    args = parser.parse_args()

    return args