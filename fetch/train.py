import argparse
import logging
import os
import string
import glob

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

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch, (freq_data, dm_data, label) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(freq_data, dm_data)
        loss = loss_fn(pred, label)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(freq_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):

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

def main():
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
        "-trn",
        "--train_data_dir",
        help="Directory containing h5 files for training.  Assumes the files contain labels",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tst",
        "--test_data_dir",
        help="Directory containing h5 files for testing.  Assumes the files contain labels",
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

    train_data = glob.glob(args.train_data_dir+ "/*.h*5")

    # Check if there's test data, if not randomly split up train
    #train_df, val_df = train_test_split(
    #    data_df, test_size=(1 - args.val_split), random_state=1993
    #)
    train_data = PulsarData(files = train_data)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    """
    test_data = PulsarData(
        list_IDs=list(val_df["h5"]),
        labels=list(val_df["label"]),
        noise=False,
        shuffle=False,
    )
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    """
    model = CompleteModel(args.model)
    print(model)
    # Need to do model.to(DEVICE) but also figure out how to get the data to DEVICE with data loader

    # Setup additional training parameters and train/test
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        #test_loop(test_dataloader, model, loss_fn)
