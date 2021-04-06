import numpy as np
import sys
import torch
from datetime import datetime

from utils import get_device, to_device, format_elapsed_time


def for_and_backward(model, batch, maps, optimizer, loss_fn):
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
def forward(model, batch, maps, loss_fn):
    """Does one forward step on given batch for validation"""
    prediction = model(batch)
    loss = loss_fn(prediction, maps)

    return loss.item(), len(batch)


@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    """Calculates average loss on the validation set"""
    results = [forward(model, batch, maps, loss_fn) for batch, *maps in val_loader]
    losses = [val[0] for val in results]
    batch_sizes = [val[1] for val in results]

    # Weighted average loss across whole dataset
    avg_loss = sum(np.multiply(losses, batch_sizes)) / sum(batch_sizes)

    return avg_loss


def fit(model, train_loader, val_loader, n_epochs, optimizer, loss_fn, output_dir, args, start_epoch=0, best_val_loss=np.inf):
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
            train_loss = for_and_backward(model, batch, maps, optimizer, loss_fn=loss_fn)

        epoch_elapsed_time = datetime.now() - epoch_start_time

        # Evaluate and persist the training progress if we hit
        # the validation interval or if this was the last epoch,
        # and if the current val_loss is better than the best
        # val_loss from all prior epochs
        if epoch % args.val_interval == 0 or epoch == max_epoch - 1:
            # Put model into evaluation mode
            model.eval()
            # Run evaluation
            val_loss = evaluate(model, val_loader, loss_fn=loss_fn)

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
