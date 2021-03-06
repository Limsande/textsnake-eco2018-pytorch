import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader

from augmentation.augmentation import RootAugmentation, RootBaseTransform
from dataloader.Eco2018Loader import DeviceLoader, Eco2018
from loss.loss import loss_fn, loss_fn2
from model.Textnet import Textnet
from model.functions import fit
from utils import get_device, get_args_parser, make_output_dir_name, print_config_file

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

    train_set = Eco2018(transformations=train_transforms)
    train_loader = DeviceLoader(
        DataLoader(
            train_set,
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

    print('Iterations per epoch:', len(train_set) // args.batch_size)

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

    if args.no_geometry_loss:
        loss_fn = loss_fn2
    else:
        loss_fn = loss_fn

    fit(model,
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        optimizer=optimizer,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        output_dir=output_dir,
        args=args,
        loss_fn=loss_fn)
