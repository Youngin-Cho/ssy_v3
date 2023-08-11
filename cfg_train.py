import argparse


def get_cfg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--look_ahead", type=int, default=2, help="number of steel plates included in states")

    # 데이터 생성 관련 파라미터
    parser.add_argument("--n_rows", type=int, default=2, help="steel plates data for storage")

    parser.add_argument("--storage", type=bool, default=True, help="steel plates data for storage")
    parser.add_argument("--reshuffle", type=bool, default=True, help="steel plates data for reshuffle")
    parser.add_argument("--retrieval", type=bool, default=True, help="steel plates data for retrieval")

    parser.add_argument("--n_bays_in_area1", type=int, default=15, help="number of bays in Area1")
    parser.add_argument("--n_bays_in_area2", type=int, default=6, help="number of bays in Area2")
    parser.add_argument("--n_bays_in_area3", type=int, default=3, help="number of bays in Area3")
    parser.add_argument("--n_bays_in_area4", type=int, default=6, help="number of bays in Area4")
    parser.add_argument("--n_bays_in_area5", type=int, default=9, help="number of bays in Area5")
    parser.add_argument("--n_bays_in_area6", type=int, default=1, help="number of bays in Area6")

    parser.add_argument("--n_from_piles_storage", type=int, default=1, help="number of from-piles in storage work")
    parser.add_argument("--n_to_piles_storage", type=int, default=5, help="number of to-piles in storage work")
    parser.add_argument("--n_from_piles_reshuffle", type=int, default=10, help="number of from-piles in reshuffle work")
    parser.add_argument("--n_to_piles_reshuffle", type=int, default=10, help="number of to-piles for in reshuffle work")
    parser.add_argument("--n_from_piles_retrieval_cn1", type=int, default=5, help="number of from-piles for conveyor 1 in retrieval work")
    parser.add_argument("--n_from_piles_retrieval_cn2", type=int, default=5, help="number of from-piles for conveyor 2 in retrieval work")
    parser.add_argument("--n_from_piles_retrieval_cn3", type=int, default=2, help="number of from-piles for trailers in retrieval work")

    parser.add_argument("--n_plates_storage", type=int, default=500, help="average number of steel plates per pile in storage work")
    parser.add_argument("--n_plates_storage", type=int, default=150, help="average number of steel plates per pile in reshuffle work")
    parser.add_argument("--n_plates_storage", type=int, default=150, help="average number of steel plates per pile in retrieval work")

    parser.add_argument("--is_crane1_working", type=bool, default=True, help="Crane-1 is working")
    parser.add_argument("--is_crane2_working", type=bool, default=True, help="Crane-2 is working")
    parser.add_argument("--safety_margin", type=int, default=5, help="safety margin between cranes")

    # 알고리즘 파라미터
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
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--lr_step", type=int, default=2500, help="Step size to reduce learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate decay ratio")
    # parser.add_argument("--base_lr", type=float, default=0.00001, help="Base learning rate")
    # parser.add_argument("--max_lr", type=float, default=0.000001, help="Maximum learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor gamma")
    # parser.add_argument("--step_size_up", type=int, default=300, help="Number of episodes increasing a learning rate")
    # parser.add_argument("--step_size_down", type=int, default=700, help="Number of episodes decreasing a learning rate")
    parser.add_argument("--tau", type=float, default=0.001, help="Soft update parameter tau")

    args = parser.parse_args()

    return args