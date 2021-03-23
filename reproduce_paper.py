import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataloader.Eco2018Loader import DeviceLoader
from dataloader.TotalTextLoader import TotalText, Augmentation, BaseTransform
from main import make_output_dir_name, print_config_file
from model.Textnet import Textnet
from model.functions import fit
from utils import get_device, get_args_parser


def create_split_indices(n: int, val_split: float) -> ([int], [int]):
    """Creates two random distinct sets of indices from [0,n)."""
    n_val = int(n * val_split)
    idx = np.random.permutation(n)
    return idx[n_val:], idx[:n_val]


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(make_output_dir_name(args), 'reproduction')
    try:
        os.makedirs(output_dir, exist_ok=True)
    except IOError as e:
        sys.exit(f'[ERROR] Could not create output directory: {e}')

    # Write configuration to log file
    try:
        print_config_file(output_dir, args)
    except IOError as e:
        sys.exit(f'[ERROR] Could not write to output directory: {e}')

    print('Running on device:', get_device())

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    train_transforms = Augmentation(size=512, mean=means, std=stds)
    val_transforms = BaseTransform(size=512, mean=means, std=stds)

    dataset = TotalText(transform=train_transforms)
    train_idx, val_idx = create_split_indices(len(dataset), val_split=0.2)

    train_loader = DeviceLoader(
        DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=SubsetRandomSampler(train_idx))

    )
    val_loader = DeviceLoader(
        DataLoader(
            dataset,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=SubsetRandomSampler(val_idx))
    )

    model = Textnet(pretrained_backbone=args.pretrained_backbone)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_val_loss = np.inf

    # Load training checkpoint from file, if requested
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f'Successfully loaded training state from {args.resume}:')
        print(f'  trained epochs: {start_epoch}')
        print(f'  best val_loss: {best_val_loss}')

    fit(model,
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        optimizer=optimizer,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        output_dir=output_dir,
        args=args)
