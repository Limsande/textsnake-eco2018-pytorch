import numpy as np
import os
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader

from augmentation.augmentation import RootAugmentation, RootBaseTransform
from dataloader.Eco2018Loader import DeviceLoader, Eco2018
from model.Textnet import Textnet
from model.functions import fit
from utils import get_device, get_args_parser


def make_output_dir_name(args):
    """Constructs a unique name for a directory in ./output using
    current time and script arguments"""
    prefix = datetime.now().strftime('%Y%m%d-%H%M')
    dir_name = f'./output/{prefix}_epochs={args.epochs}_lr={args.lr}'
    dir_name += '_with-pretrained-backbone' if args.pretrained_backbone else '_no-pretrained-backbone'
    if args.resume:
        # Extract date prefix from checkpoint path:
        # e.g. 20210320-1439 in output/20210320-1439_epochs=1_lr=0.005/checkpoint.pth
        dir_name += f'_resume={str(args.resume.parent.name).split("_")[0]}'
    return dir_name


def print_config_file(output_dir, args):
    """Prints current configuration to a file in the output directory"""
    with open(os.path.join(output_dir, 'config.cfg'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}={v}\n')
        f.write(f'device={get_device()}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    # Create output directory if it doesn't exist
    output_dir = make_output_dir_name(args)
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

    means = (77.125, 69.661, 65.885)
    stds = (9.664, 8.175, 7.810)
    train_transforms = RootAugmentation(mean=means, std=stds)
    val_transforms = RootBaseTransform(mean=means, std=stds)

    train_loader = DeviceLoader(
        DataLoader(
            Eco2018(transformations=train_transforms),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True)
    )
    val_loader = DeviceLoader(
        DataLoader(
            Eco2018(transformations=val_transforms, is_training=False),
            num_workers=args.num_workers,
            pin_memory=True)
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
