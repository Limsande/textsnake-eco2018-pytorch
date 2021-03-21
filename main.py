import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augmentation.augmentation import RootAugmentation, RootBaseTransform
from dataloader.Eco2018Loader import DeviceLoader, Eco2018
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


def forward(model, batch, maps):
    """Does one forward step on given batch for validation"""
    prediction = model(batch)
    loss = loss_fn(prediction, maps)

    return loss.item(), len(batch)


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


def loss_fn(prediction, maps):
    # TODO Online hard negative mining for tr_loss

    tr_pred = prediction[:, :2]
    tr_true = maps[0].float()
    tcl_pred = prediction[:, 2:4]
    tcl_true = maps[1].float()

    # tcl_loss only takes pixels inside text region into account.
    # We can use tr_true (Nx512x512, 1=text region, 0=no text region) to mask
    # the two channels of tcl_pred (Nx2x512x512) one by one. This lets us
    # construct a tensor tcl_pred_inside_tr with the same dimensions as tcl_pred,
    # but only with the values from tcl_pred, where tr_true is 1. Values outside
    # the text region are 0.
    tcl_pred_inside_tr = torch.stack(
            [
                torch.where(tr_true > 0, tcl_pred[:, 0], tr_true),
                torch.where(tr_true > 0, tcl_pred[:, 1], tr_true)
            ], dim=1)

    # Geometry loss only takes pixels inside tcl into account. Use tcl_true as
    # mask, like tr_true above. But this time, there is only one channel for
    # radii, sine, and cosine, each. So no need to stack.
    r_pred_inside_tcl = torch.where(tcl_true > 0, prediction[:, 4], tcl_true)
    r_true = maps[2]
    cos_pred_inside_tcl = torch.where(tcl_true > 0, prediction[:, 5], tcl_true)
    cos_true = maps[3]
    sin_pred_inside_tcl = torch.where(tcl_true > 0, prediction[:, 6], tcl_true)
    sin_true = maps[4]

    tr_loss = F.cross_entropy(tr_pred, tr_true.long())
    tcl_loss = F.cross_entropy(tcl_pred_inside_tr, tcl_true.long())
    radii_loss = F.smooth_l1_loss(r_pred_inside_tcl, r_true)
    sin_loss = F.smooth_l1_loss(sin_pred_inside_tcl, sin_true)
    cos_loss = F.smooth_l1_loss(cos_pred_inside_tcl, cos_true)

    return tr_loss + tcl_loss + radii_loss + sin_loss + cos_loss


def make_output_dir_name():
    """Constructs a unique name for a directory in ./output using
    current time and script arguments"""
    prefix = datetime.now().strftime('%Y%m%d-%H%M')
    dir_name = f'./output/{prefix}_epochs={args.epochs}_lr={args.lr}'
    if args.resume:
        # Extract date prefix from checkpoint path:
        # e.g. 20210320-1439 in output/20210320-1439_epochs=1_lr=0.005/checkpoint.pth
        dir_name += f'_resume={str(args.resume.parent.name).split("_")[0]}'
    return dir_name


def format_elapsed_time(delta) -> str:
    # Remove microseconds
    return str(delta).split('.')[0]



if __name__ == '__main__':
    args = get_args_parser().parse_args()

    # Create output directory if it doesn't exist
    output_dir = make_output_dir_name()
    try:
        os.makedirs(output_dir, exist_ok=True)
    except IOError as e:
        sys.exit(f'[ERROR] Could not create output directory: {e}')

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

    model = Textnet()
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
