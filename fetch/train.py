import argparse
import logging
import os
import string
import glob

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from torcheval.metrics.functional import binary_precision, binary_recall, binary_f1_score

from fetch.pulsar_data import PulsarData
from fetch.model import PulsarModel

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)
LOGGINGFORMAT = ("%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def train_loop(dataloader: DataLoader, 
               model: nn.Module, 
               loss_fn, 
               optimizer,
               batch_size: int) -> None:
    r"""
    """
    size = len(dataloader.dataset)
    
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch, (freq_data, dm_data, label) in enumerate(dataloader): 
        
        freq_data = freq_data.to(DEVICE)
        dm_data = dm_data.to(DEVICE)
        label = label.to(DEVICE)

        # Compute prediction and loss
        pred = model(freq_data, dm_data)
        loss = loss_fn(pred, label)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(freq_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)

def evaluate_loop(dataloader: DataLoader, model: nn.Module, loss_fn) -> None:
    r"""
    """

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for freq_data, dm_data, label in dataloader:

            freq_data = freq_data.to(DEVICE)
            dm_data = dm_data.to(DEVICE)
            label = label.to(DEVICE)

            pred = model(freq_data, dm_data)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n", flush=True)

def test_model(dataloader: DataLoader, model: nn.Module) -> None:
    r"""
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    truth = torch.empty(0)
    predictions = torch.empty(0)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for freq_data, dm_data, label in dataloader:

            truth = torch.cat((truth, label))

            freq_data = freq_data.to(DEVICE)
            dm_data = dm_data.to(DEVICE)
            label = label.to(DEVICE)

            # Get predictions from model and move to CPU
            pred = model(freq_data, dm_data)
            pred = pred.to('cpu')

            print(f"pred shape: {pred.shape}")
            print(f"predictions shape: {predictions.shape}")
            predictions = torch.cat((predictions, pred))
            
    recall = binary_recall(predictions, truth)
    precision = binary_precision(predictions, truth)
    f1 = binary_f1_score(predictions, truth)

    print(f"--- Test results ---\n Recall: {recall}\n Precision: {precision}\n F1: {f1}\n", flush=True)

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
        default=None,
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

    logging_format = ("%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    if args.model not in list(string.ascii_lowercase)[:11]:
        raise ValueError(f"Model only range from a -- j.")

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    logger.info(f"Using {DEVICE} for computation")

    # Load training and split 85% to 15% into train/validate
    train_data_files = glob.glob(args.train_data_dir + "/*.h*5")
    train_data = PulsarData(files=train_data_files, noise=True)
    train_data, validate_data = random_split(train_data, [0.85, 0.15])

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=args.batch_size, shuffle=False)

    # Add some noise to freq data to help avoid overtraining
    logger.info(f"Adding noise to training data")
    for freq_data, dm_data, label in train_dataloader:
        freq_data += torch.normal(0.0, 1.0, size=freq_data.shape)

    # Build the model and push it to proper compute device
    model = PulsarModel(args.model).to(DEVICE)

    # Setup additional training parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    # Train the model
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------", flush=True)
        train_loop(train_dataloader, model, loss_fn, optimizer, args.batch_size)
        evaluate_loop(validate_dataloader, model, loss_fn)

    # Test the trained model
    if args.test_data_dir is not None:
        test_data_files = glob.glob(args.test_data_dir + "/*.h*5")
        test_data = PulsarData(files=test_data_files)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        test_model(test_dataloader, model)

    # Save the model weights
    weight_file = "/model_" + args.model + "_weights.pth"
    torch.save(model.state_dict(), args.output_path + weight_file)
