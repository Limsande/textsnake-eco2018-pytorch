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
    results = [forward(model, batch, torch.stack(maps, dim=1)) for batch, *maps in val_loader]
    losses = [val[0] for val in results]
    batch_sizes = [val[1] for val in results]

    # Weighted average loss across whole dataset
    avg_loss = sum(np.multiply(losses, batch_sizes)) / sum(batch_sizes)

    return avg_loss


def fit(model, train_loader, val_loader, n_epochs, optimizer):
    """Fits model to given data"""

    # Prepare csv file for logging the loss
    loss_file = f'./{output_dir}/loss.csv'
    try:
        with open(loss_file, 'w') as f:
            f.write('Epoch;Train_loss;Val_loss')
    except IOError as e:
        print(f'[WARNING] Could not create loss file {loss_file}:', e, file=sys.stderr)

    to_device(model, get_device())

    # This is the training process
    for epoch in range(n_epochs):

        # Put model into training mode
        model.train()
        # Train one epoch
        for batch, *maps in train_loader:
            # TODO stack necessary?
            maps = torch.stack(maps, dim=1)
            train_loss = for_and_backward(model, batch, maps, optimizer)

        # Evaluate if we hit the validation interval
        # or if this was the last epoch
        val_loss = None
        if epoch % args.val_interval == 0 or epoch == args.n_epochs - 1:
            # Put model into evaluation mode
            model.eval()
            # Run evaluation
            val_loss = evaluate(model, val_loader)

        if val_loss:
            print(f'Epoch {epoch}: Train loss={train_loss}; Val loss={val_loss}')
        else:
            print(f'Epoch {epoch}: Train loss={train_loss}')

        try:
            with open(loss_file, 'a') as f:
                f.write(f'\n{epoch};{train_loss};{val_loss}')
        except IOError as e:
            print(f'[WARNING] Could not write to loss file:', e, file=sys.stderr)


def loss_fn(prediction, maps):
    # TODO Online hard negative mining for tr_loss

    # TODO tcl_loss only inside tr

    # TODO tr and tcl are probs along dim 1
    tr_logits = prediction[:, :2]
    #tr_pred = (tr_logits[:, 1] > tr_logits[:, 0]).float()
    tr_pred = tr_logits

    # NxHxW --> Nx(H*W)
    #tr_pred = tr_pred.view((tr_pred.shape[0], tr_pred.shape[1] * tr_pred.shape[2]))
    #tr_true = maps[:, 0].view((maps[:, 0].shape[0], maps[:, 0].shape[1] * maps[:, 0].shape[2]))
    # TODO move type cast somewhere better
    tr_true = maps[:, 0].long()

    tcl_logits = prediction[:, 2:4]
    #tcl_pred = (tcl_logits[:, 1] > tcl_logits[:, 0]).float()
    tcl_pred = tcl_logits

    # NxHxW --> Nx(H*W)
    #tcl_pred = tcl_pred.view((tcl_pred.shape[0], tcl_pred.shape[1] * tcl_pred.shape[2]))
    #tcl_true = maps[:, 1].view((maps[:, 1].shape[0], maps[:, 1].shape[1] * maps[:, 1].shape[2]))
    tcl_true = maps[:, 1].long()

    r_pred = prediction[:, 4]
    r_true = maps[:, 2]
    cos_pred = prediction[:, 5]
    cos_true = maps[:, 3]
    sin_pred = prediction[:, 6]
    sin_true = maps[:, 4]

    tr_loss = F.cross_entropy(tr_pred, tr_true)
    tcl_loss = F.cross_entropy(tcl_pred, tcl_true)
    radii_loss = F.smooth_l1_loss(r_pred, r_true)
    sin_loss = F.smooth_l1_loss(sin_pred, sin_true)
    cos_loss = F.smooth_l1_loss(cos_pred, cos_true)

    return tr_loss + tcl_loss + radii_loss + sin_loss + cos_loss


def make_output_dir_name():
    """Constructs a unique name for a directory in ./output using
    current time and script arguments"""
    prefix = datetime.now().strftime('%Y%m%d-%H%M')
    dir_name = f'./output/{prefix}_epochs={args.epochs}_lr={args.lr}'
    return dir_name


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    # Create output directory if it doesn't exist
    output_dir = make_output_dir_name()
    try:
        os.makedirs(output_dir, exist_ok=True)
    except IOError as e:
        sys.exit(f'[ERROR] Could not create output directory: {e}')

    print(get_device())

    means = (77.125, 69.661, 65.885)
    stds = (9.664, 8.175, 7.810)
    train_transforms = RootAugmentation(mean=means, std=stds)
    val_transforms = RootBaseTransform(mean=means, std=stds)

    data_root = 'data/Eco2018-Test'
    train_loader = DeviceLoader(
        DataLoader(Eco2018(data_root=data_root, transformations=train_transforms), shuffle=True, batch_size=1))
    val_loader = DeviceLoader(
        DataLoader(Eco2018(data_root=data_root, transformations=val_transforms, is_training=False)))

    model = Textnet()
    # for batch, *maps in train_loader:
    #     out = model(batch)
    #     print(out.shape)
    #     break

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fit(model, train_loader, val_loader, n_epochs=args.epochs, optimizer=optimizer)
