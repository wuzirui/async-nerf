import configargparse


def get_opts_base():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)
    parser.add_argument('--train_datapath', required=True, type=str, help='input trajectories text')
    parser.add_argument('--val_timestamp_file', required=False, type=str, help='filepath to collection of validation timestamps')
    parser.add_argument('--exp_name', type=str, required=True, help='path to experiment logs storage')
    parser.add_argument('--dataset_type', choices=['kitti'], default='kitti')

    parser.add_argument('--n_layers', type=int, default=8, help='num of hidden layers in multi-layer perceptron')
    parser.add_argument('--n_channels', type=int, default=256, help='num of neurons in each hidden layer')
    parser.add_argument('--skip_connections', type=int, nargs='*', default=[4], help='skip connections like nerf')
    parser.add_argument('--random_seed', type=int, default=28, help='random seeds for rngs')
    return parser
