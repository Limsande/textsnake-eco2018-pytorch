import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentation.augmentation import RootAugmentation, RootBaseTransform
from dataloader.Eco2018Loader import DeviceLoader, Eco2018
from loss.loss import loss_fn
from model.Textnet import Textnet
from utils import get_device, to_device, get_args_parser


def for_and_backward(model, batch, maps, optimizer):
    """Does one forward and one backward step using the given batch"""
    prediction = model(batch)
    loss = loss_fn(prediction, maps)
    # Calculate gradients
    loss.backward()
    # Update model parameters from gradients
    optimizer.step()
    # Reset gradients
    optimizer.zero_grad()

    return loss


@torch.no_grad()
def forward(model, batch, maps):
    """Does one forward step on given batch for validation"""
    prediction = model(batch)
    loss = loss_fn(prediction, maps)

    return loss.item(), len(batch)


@torch.no_grad()
def evaluate(model, val_loader):
    """Calculates average loss on the validation set"""
    results = [forward(model, batch, maps) for batch, *maps in val_loader]
    losses = [val[0] for val in results]
    batch_sizes = [val[1] for val in results]

    # Weighted average loss across whole dataset
    avg_loss = sum(np.multiply(losses, batch_sizes)) / sum(batch_sizes)

    return avg_loss


def fit(model, train_loader, val_loader, n_epochs, optimizer, start_epoch=0, best_val_loss=np.inf):
    """Fits model to given data"""

    # Save training checkpoints here
    checkpoint_file = f'{output_dir}/checkpoint.pth'

    # Prepare csv file for logging the loss
    loss_file = f'{output_dir}/loss.csv'
    try:
        with open(loss_file, 'w') as f:
            f.write('Epoch;Train_loss;Val_loss')
    except IOError as e:
        print(f'[WARNING] Could not create loss file {loss_file}:', e, file=sys.stderr)

    to_device(model, get_device())

    # This is the training process
    max_epoch = start_epoch + n_epochs
    start_time = datetime.now()
    print(start_time.strftime("Started on %a, %d.%m.%Y at %H:%M"))
    for epoch in range(start_epoch, max_epoch):
        epoch_start_time = datetime.now()

        # Put model into training mode
        model.train()
        # Train one epoch
        for batch, *maps in train_loader:
            train_loss = for_and_backward(model, batch, maps, optimizer)

        epoch_elapsed_time = datetime.now() - epoch_start_time

        # Evaluate and persist the training progress if we hit
        # the validation interval or if this was the last epoch,
        # and if the current val_loss is better than the best
        # val_loss from all prior epochs
        if epoch % args.val_interval == 0 or epoch == max_epoch - 1:
            # Put model into evaluation mode
            model.eval()
            # Run evaluation
            val_loss = evaluate(model, val_loader)

            print(f'Epoch {epoch + 1}: Train loss={train_loss}; Val loss={val_loss};',
                  f'Time elapsed in this epoch: {format_elapsed_time(epoch_elapsed_time)}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_file)

                print(f'Saved checkpoint in {checkpoint_file}')
        else:
            print(f'Epoch {epoch + 1}: Train loss={train_loss}')

        try:
            with open(loss_file, 'a') as f:
                f.write(f'\n{epoch};{train_loss};{val_loss}')
        except IOError as e:
            print(f'[WARNING] Could not write to loss file:', e, file=sys.stderr)

    total_elapsed_time = datetime.now() - start_time
    print('Total elapsed time:', format_elapsed_time(total_elapsed_time))


def make_output_dir_name():
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


def format_elapsed_time(delta) -> str:
    # Remove microseconds
    return str(delta).split('.')[0]


def print_config_file(output_dir):
    """Prints current configuration to a file in the output directory"""
    with open(os.path.join(output_dir, 'config.cfg'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}={v}\n')
        f.write(f'device={get_device()}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    # Create output directory if it doesn't exist
    output_dir = make_output_dir_name()
    try:
        os.makedirs(output_dir, exist_ok=True)
    except IOError as e:
        sys.exit(f'[ERROR] Could not create output directory: {e}')

    # Write configuration to log file
    try:
        print_config_file(output_dir)
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

    fit(model,
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        optimizer=optimizer,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss)
