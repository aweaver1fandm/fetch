#!/usr/bin/env python3

import argparse
import logging
import os
import string

import pandas as pd
from sklearn.model_selection import train_test_split

from fetch.pulsar_data import PulsarData
from fetch.utils import get_model
from fetch.utils import ready_for_train

logger = logging.getLogger(__name__)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)"
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-g", "--gpu_id", help="GPU ID", type=int, required=False, default=0
    )
    parser.add_argument(
        "-n", "--nproc", help="Number of processors for training", default=4, type=int
    )
    parser.add_argument(
        "-c",
        "--data_csv",
        help="CSV with candidate h5 paths and labels",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size for training data", default=8, type=int
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of epochs for training", default=15, type=int
    )
    parser.add_argument(
        "-p",
        "--patience",
        help="Layer patience, stop training if validation loss does not decreate",
        default=3,
        type=int,
    )
    parser.add_argument(
        "-nft",
        "--n_ft_layers",
        help="Number of layers in FT model to train",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-ndt",
        "--n_dt_layers",
        help="Number of layers in DT model to train",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-nf",
        "--n_fusion_layers",
        help="Number of layers to train post FT and DT models",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Place to save the weights and training logs",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-vs",
        "--val_split",
        help="Percent of data to use for validation",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "-m", "--model", help="Index of the model to train", required=True, type=str
    )
    args = parser.parse_args()

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    if args.model not in list(string.ascii_lowercase)[:11]:
        raise ValueError(f"Model only range from a -- j.")

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    if args.n_fusion_layers >= 9:
        raise ValueError(
            f"Cannot open {args.n_fusion_layers} for training. Models only have 6 layers after FT and DT models."
        )

    data_df = pd.read_csv(args.data_csv)

    train_df, val_df = train_test_split(
        data_df, test_size=(1 - args.val_split), random_state=1993
    )
    train_data = PulsarData(
        list_IDs=list(train_df["h5"]),
        labels=list(train_df["label"]),
        noise=True,
    )
    train_dataloader = DataLoader(train_data, batch_size=32,shuffle=True)

    test_data = PulsarData(
        list_IDs=list(val_df["h5"]),
        labels=list(val_df["label"]),
        noise=False,
        shuffle=False,
    )
    test_dataloader = DataLoader(test_data, batch_size=32,shuffle=False)

    model_to_train = get_model(args.model)

    model_to_train = ready_for_train(
        model_to_train,
        ndt=args.n_dt_layers,
        nft=args.n_ft_layers,
        nf=args.n_fusion_layers,
    )

    trained_model, history = train(
        model_to_train,
        epochs=args.epochs,
        patience=args.patience,
        output_path=args.output_path,
        nproc=args.nproc,
        train_obj=train_data_generator,
        val_obj=validate_data_generator,
    )
