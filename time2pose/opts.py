import configargparse


def get_opts_base():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)
    parser.add_argument('--datapath', required=True, type=str, help='input trajectories text')
    parser.add_argument('--test_datapath', required=False, type=str, help='input trajectories for inference')
    parser.add_argument('--exp_name', type=str, required=True, help='path to experiment logs storage')
    parser.add_argument('--dataset_type', choices=['default'], default='default')
    parser.add_argument('--n_val', type=int, default=20, help='num of validation frames')
    parser.add_argument('--pose_scale_factor', type=float, default=200)

    parser.add_argument('--start_timestamp', type=float, default=0)
    parser.add_argument('--centroid', type=float, nargs=3, default=[0., 0., 0.])

    parser.add_argument('--translation_weight', type=float, default=1, help='weight of translation loss')

    parser.add_argument('--n_layers', type=int, default=9, help='num of hidden layers in multi-layer perceptron')
    parser.add_argument('--n_channels', type=int, default=512, help='num of neurons in each hidden layer')
    parser.add_argument('--skip_connections', type=int, nargs='*', default=[4], help='skip connections like nerf')
    parser.add_argument('--t_embedding_freq', type=int, default=0, help='positional encoding for timestamps')
    parser.add_argument('--random_seed', type=int, default=28, help='random seeds for rngs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--train_epochs', type=int, default=100000)
    parser.add_argument('--val_interval', type=int, default=1000)
    return parser
