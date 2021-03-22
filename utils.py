import pathlib
from argparse import ArgumentParser

import torch
import torch.nn.functional as F


def softmax_and_regularize(pseudo_predictions):
    # softmax on text regions and center lines
    output = torch.zeros_like(pseudo_predictions)
    output[:, :2] = F.softmax(pseudo_predictions[:, :2], dim=1)  # text regions
    output[:, 2:4] = F.softmax(pseudo_predictions[:, 2:4], dim=1)  # text center lines

    output[:, 4] = pseudo_predictions[:, 4]  # radii

    # regularizing cosθ and sinθ so that the squared sum equals 1
    scale = torch.sqrt(1. / (torch.pow(pseudo_predictions[:, 5], 2) + torch.pow(pseudo_predictions[:, 6], 2)))
    output[:, 5] = pseudo_predictions[:, 5] * scale  # cosine
    output[:, 6] = pseudo_predictions[:, 6] * scale  # sine

    return output


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
    parser.add_argument(
        '--pretrained-backbone',
        action='store_true',
        help='Use as backbone a VGG16 pretrained on ImageNet from the torchvision GitHub repo'
    )

    return parser
