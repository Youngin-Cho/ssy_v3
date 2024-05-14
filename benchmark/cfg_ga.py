import argparse


def get_cfg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="model file path")
    parser.add_argument("--param_path", type=str, default=None, help="hyper-parameter file path")
    parser.add_argument("--data_dir", type=str, default=None, help="test data path")
    parser.add_argument("--log_dir", type=str, default=None, help="log file path")

    parser.add_argument("--n_rows", type=int, default=2, help="steel plates data for storage")
    parser.add_argument("--n_bays_in_area1", type=int, default=15, help="number of bays in Area1")
    parser.add_argument("--n_bays_in_area2", type=int, default=6, help="number of bays in Area2")
    parser.add_argument("--n_bays_in_area3", type=int, default=3, help="number of bays in Area3")
    parser.add_argument("--n_bays_in_area4", type=int, default=6, help="number of bays in Area4")
    parser.add_argument("--n_bays_in_area5", type=int, default=9, help="number of bays in Area5")
    parser.add_argument("--n_bays_in_area6", type=int, default=1, help="number of bays in Area6")

    parser.add_argument("--is_crane1_working", type=int, default=1, help="Crane-1 is working")
    parser.add_argument("--is_crane2_working", type=int, default=1, help="Crane-2 is working")
    parser.add_argument("--safety_margin", type=int, default=5, help="safety margin between cranes")

    parser.add_argument("--multi_num", type=int, default=3, help="Number of plates allowed for multi-loading")
    parser.add_argument("--multi_w", type=int, default=20.0, help="Total weight of plates allowed for multi-loading")
    parser.add_argument("--multi_dis", type=int, default=2, help="Distance allowed for multi-loading")

    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--record_events", type=int, default=0, help="Whether to record events")

    args = parser.parse_args()

    return args