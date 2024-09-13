import argparse


def get_cfg():

    parser = argparse.ArgumentParser(description="")

    # 데이터 생성 관련 파라미터
    parser.add_argument("--n_rows", type=int, default=2, help="steel plates data for storage")

    parser.add_argument("--n_bays_in_area1", type=int, default=15, help="number of bays in Area1")
    parser.add_argument("--n_bays_in_area2", type=int, default=6, help="number of bays in Area2")
    parser.add_argument("--n_bays_in_area3", type=int, default=3, help="number of bays in Area3")
    parser.add_argument("--n_bays_in_area4", type=int, default=6, help="number of bays in Area4")
    parser.add_argument("--n_bays_in_area5", type=int, default=9, help="number of bays in Area5")
    parser.add_argument("--n_bays_in_area6", type=int, default=1, help="number of bays in Area6")

    parser.add_argument("--is_crane1_working", type=bool, default=True, help="Crane-1 is working")
    parser.add_argument("--is_crane2_working", type=bool, default=True, help="Crane-2 is working")
    parser.add_argument("--safety_margin", type=int, default=5, help="safety margin between cranes")

    parser.add_argument("--multi_num", type=int, default=3, help="Number of plates allowed for multi-loading")
    parser.add_argument("--multi_w", type=float, default=20.0, help="Total weight of plates allowed for multi-loading")
    parser.add_argument("--multi_dis", type=int, default=2, help="Distance allowed for multi-loading")

    parser.add_argument("--look_ahead", type=int, default=3, help="number of steel plates included in terminals")
    parser.add_argument("--pop_size", type=int, default=60, help="population size")
    parser.add_argument("--min_depth", type=int, default=10, help="minimal initial random tree depth")
    parser.add_argument("--max_depth", type=int, default=15, help="maximal initial random tree depth")
    parser.add_argument("--generations", type=int, default=50, help="maximal number of generations to run evolution")
    parser.add_argument("--tournament_size", type=int, default=10, help="size of tournament for tournament selection")
    parser.add_argument("--xo_rate", type=float, default=0.8, help="crossover rate")
    parser.add_argument("--prob_mutation", type=float, default=0.2, help="per-node mutation probability")

    parser.add_argument("--data_dir", type=str, default="./input/gp/instances/5-10-10/", help="directory where the data for fitness calculation are stored")

    return parser.parse_args()