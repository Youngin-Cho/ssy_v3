import argparse


def get_cfg():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--vessl", type=int, default=1, help="whether to use vessl (0: False, 1:True)")

    parser.add_argument("--n_episode", type=int, default=10000, help="number of episodes")
    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--look_ahead", type=int, default=3, help="number of steel plates included in states")
    parser.add_argument("--record_events", type=int, default=0, help="Whether to record events")

    # 데이터 생성 관련 파라미터
    parser.add_argument("--n_rows", type=int, default=2, help="steel plates data for storage")

    parser.add_argument("--storage", type=int, default=1, help="steel plates data for storage")
    parser.add_argument("--reshuffle", type=int, default=1, help="steel plates data for reshuffle")
    parser.add_argument("--retrieval", type=int, default=1, help="steel plates data for retrieval")

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
    parser.add_argument("--n_plates_reshuffle", type=int, default=150, help="average number of steel plates per pile in reshuffle work")
    parser.add_argument("--n_plates_retrieval", type=int, default=150, help="average number of steel plates per pile in retrieval work")

    parser.add_argument("--is_crane1_working", type=bool, default=True, help="Crane-1 is working")
    parser.add_argument("--is_crane2_working", type=bool, default=True, help="Crane-2 is working")
    parser.add_argument("--safety_margin", type=int, default=5, help="safety margin between cranes")

    parser.add_argument("--multi_num", type=int, default=3, help="Number of plates allowed for multi-loading")
    parser.add_argument("--multi_w", type=float, default=20.0, help="Total weight of plates allowed for multi-loading")
    parser.add_argument("--multi_dis", type=int, default=2, help="Distance allowed for multi-loading")

    parser.add_argument("--use_gnn", type=int, default=1, help="whether to use gnn")
    parser.add_argument("--embed_dim", type=int, default=128, help="node embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="multi-head attention in HGT layers")
    parser.add_argument("--num_HGT_layers", type=int, default=2, help="number of HGT layers")
    parser.add_argument("--num_q_layers", type=int, default=2, help="number of state-action feature layers")
    parser.add_argument("--n_cos", type=int, default=64, help="dimension of the cosine basis used to embed quantile fractions")
    parser.add_argument("--num_quantiles", type=int, default=8, help="number of quantile samples used for training and action selection")
    parser.add_argument("--kappa", type=float, default=1.0, help="threshold of the quantile huber loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="learning rate decay ratio")
    parser.add_argument("--lr_step", type=int, default=2000, help="step size to reduce learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount ratio")
    parser.add_argument("--buffer_size", type=int, default=50000, help="replay buffer size")
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--min_buffer_size", type=int, default=1000, help="minimum buffer size before training starts")
    parser.add_argument("--train_every", type=int, default=10, help="the number of steps between training updates")
    parser.add_argument("--target_update", type=int, default=200, help="target network sync period (in training steps)")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="initial epsilon for epsilon-greedy")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="minimum epsilon for epsilon-greedy")
    parser.add_argument("--epsilon_decay", type=float, default=0.9995, help="epsilon decay ratio per training step")

    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=1000, help="Save a model every x episodes")
    parser.add_argument("--new_instance_every", type=int, default=10, help="Generate new scenarios every x episodes")

    parser.add_argument("--val_dir", type=str, default=None, help="directory where the validation data are stored")

    return parser.parse_args()
