import pathlib
from argparse import ArgumentParser

import torch


def to_device(data, device):
    """Loads given data into the RAM of given device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_device() -> torch.device:
    """Return a torch.device of type 'cuda' if available, else of type 'cpu'"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='',
        epilog=''
    )

    parser.add_argument(
        '--lr',
        type=float,
        required=True,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        metavar='N',
        required=True,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        metavar='N',
        default=1,
        help='Batch size (default 1)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        metavar='N',
        default=8,
        help='Number of processes used to load data'
    )
    parser.add_argument(
        '--val-interval',
        type=int,
        metavar='N',
        default=1,
        help='Evaluate model after each N epochs (default 1)'
    )
    parser.add_argument(
        '--resume',
        type=pathlib.Path,
        metavar='FILE',
        help='Resume training at checkpoint loaded from FILE'
    )

    return parser
