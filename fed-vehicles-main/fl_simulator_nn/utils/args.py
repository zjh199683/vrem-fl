import os
import argparse


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    args_string = ""

    args_to_show = ["experiment", "method"]
    for arg in args_to_show:
        args_string = os.path.join(args_string, str(getattr(args, arg)))

    return args_string


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'experiment',
        help='name of experiment',
        type=str
    )
    parser.add_argument(
        'method',
        help='the method to be used;'
             ' possible are `local`, `FedAvg` and `FedNova`',
        type=str
    )
    parser.add_argument(
        'selection_strategy',
        help='strategy for the selection of clients;'
             'possible are `optimal`, `aoi`, `random`, `round_robin`, `channel_gain`',
        type=str
    )
    parser.add_argument(
        '--cpu_strategy',
        help='strategy for the selection of computation slots;'
             'possible are `opt`, `min`, `max`',
        type=str,
        default='opt'
    )
    parser.add_argument(
        '--tx_strategy',
        help='strategy for the selection of transmission slots;'
             'possible are `opt`, `min_lat`, `min_tx_time`',
        type=str,
        default='opt'
    )
    parser.add_argument(
        '--sim_len',
        help='Total length of the simulation (steps)',
        type=int,
        default=3600
    )
    parser.add_argument(
        '--min_time',
        help='Number of steps to discard at the beginning',
        type=int,
        default=600
    )
    parser.add_argument(
        '--sampling_rate',
        help='proportion of clients to be used at each round; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--n_clients',
        help='Fixed number of clients to sample at each round. (default 30)',
        type=int,
        default=30
    )
    parser.add_argument(
        '--input_dimension',
        help='the dimension of one input sample; only used for synthetic datasets',
        type=int,
        default=None
    )
    parser.add_argument(
        '--time_slot',
        help='Duration of a time slot (s)',
        type=float,
        default=1
    )
    parser.add_argument(
        '--max_latency',
        help='Number of slots to wait at most before aggregating the model',
        type=int,
        default=100
    )
    parser.add_argument(
        '--output_dimension',
        help='the dimension of output space; only used for synthetic datasets',
        type=int,
        default=None
    )
    parser.add_argument(
        '--n_rounds',
        help='number of communication rounds; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--bz',
        help='batch_size; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--fit_epoch',
        help='if selected, one local step corresponds to one epoch, otherwise it correspond to one mini batch',
        action='store_true'
    )
    parser.add_argument(
        '--min_local_steps',
        help='minimal number of local local_steps allowed before communication; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--max_local_steps',
        help='maximal number of local local_steps allowed before communication; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log_freq',
        help='frequency of writing logs; defaults is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or cuda; default is cpu',
        type=str,
        default="cpu"
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer to be used for the training; default is sgd',
        type=str,
        default="sgd"
    )
    parser.add_argument(
        "--local_lr",
        type=float,
        help='learning rate at local client; default is 1e-3',
        default=1e-3
    )
    parser.add_argument(
        "--server_lr",
        type=float,
        help='learning rate at server; default is 1e-3',
        default=1e-3
    )
    parser.add_argument(
        "--fix_server_lr",
        help='if selected the server learning rate is fixed (equal to `server_lr`) '
             'instead of being adapted according to the proposed strategy;',
        action='store_true'
    )
    parser.add_argument(
        "--max_server_lr",
        help='if selected the server learning rate is fixed and equal to the weighted maximum number of steps '
             'instead of being adapted according to the proposed strategy;',
        action='store_true'
    )
    parser.add_argument(
        "--lr_scheduler",
        help='learning rate decay scheme to be used;'
             'possible are "sqrt", "linear", "cosine_annealing" and "constant"(no learning rate decay);'
             'default is "constant"',
        type=str,
        default="constant"
    )
    parser.add_argument(
        "--use_float64",
        help='if selected a 64-bits representation is used, this will roughly double memory consumption;',
        action='store_true'
    )
    parser.add_argument(
        "--mu",
        help='proximal term weight, only used when --optimizer=`prox_sgd`; default is `0.`',
        type=float,
        default=0
    )
    parser.add_argument(
        '--validation',
        help='if chosen the validation part will be used instead of test part;'
             ' make sure to use `val_frac > 0` in `generate_data.py`;',
        action='store_true'
    )
    parser.add_argument(
        "--verbose",
        help='verbosity level, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`;',
        type=int,
        default=0
    )
    parser.add_argument(
        "--logs_save_path",
        help='path to write logs; if not passed, it is set using arguments',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--gradients_save_path",
        help="path to save full gradients of each round, expected to be a path to a `.h5` file;"
             "if not specified, gradients are not saved",
        default=None
    )
    parser.add_argument(
        "--metadata_save_path",
        help="path to save metadata, i.e., full gradients norm, server learning rate "
             "and the number of local local_steps, for each round, expected to be a path to a `.json` file;"
             "if not specified, metadata is not saved",
        default=None
    )
    parser.add_argument(
        "--cfg_file_path",
        help="path to configuration, expected to be a JSON file,",
        default=None
    )
    parser.add_argument(
        "--chkpts_save_path",
        help='directory to save checkpoints once the training is over; if not specified checkpoints are not saved',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--seed",
        help='random seed',
        type=int,
        default=1234
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args
