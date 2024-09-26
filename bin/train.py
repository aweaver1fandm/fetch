#!/usr/bin/env python3

import argparse
import logging
import os
import string

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from fetch.pulsar_data import PulsarData
from fetch.model import CompleteModel

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)
LOGGINGFORMAT = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    # Set the model to training mode - important for batch normalization and dropout layers
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
        "-b", "--batch_size", help="Batch size for training data", default=32, type=int
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of epochs for training", default=15, type=int
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Place to save the weights and training logs",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m", "--model", help="Index of the model to train", required=True, type=str
    )
    args = parser.parse_args()
    parser.add_argument(
        "-lr", "--learning_rate", help="Training learning rate", default=1e-3, type=float
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

    # Stuff I'll need
    #model = CNN_3()
    #criterion = CrossEntropyLoss()
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    data_df = pd.read_csv(args.data_csv)

    train_df, val_df = train_test_split(
        data_df, test_size=(1 - args.val_split), random_state=1993
    )
    train_data = PulsarData(
        list_IDs=list(train_df["h5"]),
        labels=list(train_df["label"]),
        noise=True,
    )
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = PulsarData(
        list_IDs=list(val_df["h5"]),
        labels=list(val_df["label"]),
        noise=False,
        shuffle=False,
    )
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = CompleteModel(args.model)

    # Setup additional training parameters and train/test
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
