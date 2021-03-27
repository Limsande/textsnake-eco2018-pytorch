import numpy as np
import pathlib
import torch
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser


def format_elapsed_time(delta) -> str:
    # Remove microseconds
    return str(delta).split('.')[0]


def load_img_as_np_array(path):
    return np.array(Image.open(path))


def softmax(pseudo_predictions):
    """Apply softmax to text regions and center lines in
    pseudo_predictions[:, :2] and [:, 2:4], respectively."""
    output = pseudo_predictions.clone()
    output[:, :2] = F.softmax(pseudo_predictions[:, :2], dim=1)
    output[:, 2:4] = F.softmax(pseudo_predictions[:, 2:4], dim=1)
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
