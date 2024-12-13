import argparse
import os
import string
import glob
import sys
import numpy as np
from shutil import copy

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from torcheval.metrics.functional import binary_precision, binary_recall, binary_f1_score

from fetch.pulsar_data import PulsarData, printObsCounts
from fetch.model import TorchvisionModel

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
def train_loop(dataloader: DataLoader, 
               model: nn.Module,
               data: str,
               loss_fn: _Loss, 
               optimizer: Optimizer,
               batch_size: int,
    ) -> None:
    r"""Perform a single pass of training on a model

    Args:
        dataloader: Contains batches of data
        model: The model being used
        data: The type of data being used for training freq or dm
        loss_fn: Loss function used for training
        optimizer: Optimization being used for training
        batch_size: Number of data points per batch
    """

    size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    for batch_idx, (freq_data, dm_data, labels) in enumerate(dataloader):
        batch_data = None

        # Load labels to device
        labels = labels.to(DEVICE, non_blocking=True)

        # Add some noise to freq data to help avoid overtraining
        if data == "freq":
            noise = torch.randn_like(freq_data) * .1
            batch_data = freq_data + noise
        elif data == "dm":
            batch_data = dm_data
        else:
            print(f"Invalid data type provided: {data}", flush=True)
            sys.exit(0)

        batch_data = batch_data.to(DEVICE, non_blocking=True)
        predicted = model(batch_data)

        # Compute loss and backpropogate
        loss = loss_fn(predicted, labels.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            loss = loss.item() 
            current = batch_idx * batch_size + len(freq_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    
def validate_loop(dataloader: DataLoader, 
                  model: nn.Module, 
                  data: str,
                  loss_fn: _Loss,
                  prob: float,
    ) -> float:
    r""" Performs a single validation pass for a model

    Args:
        dataloader: Contains batches of data
        model: The model being used
        data: The type of data being used for training freq or dm
        loss_fn: Loss function used for training
        prob: Probability criteria to determine if observation is pulsar or not
    
    Returns:
        The validation loss for this pass
    """

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, correct = 0, 0

    # Used to calculate F1
    truth = []
    predictions = []

    # Evaluating the model with torch.no_grad() ensures 
    # that no gradients are computed during validation
    with torch.no_grad():
        for batch_idx, (freq_data, dm_data, labels) in enumerate(dataloader):
            batch_data = None

            # Load labels to device
            labels = labels.to(DEVICE, non_blocking=True)

            # Load data to device and make predictions
            if data == "freq":
                batch_data = freq_data
            elif data == "dm":
                batch_data = dm_data
            else:
                print(f"Invalid data type provided: {data}")
                sys.exit(0)

            batch_data = batch_data.to(DEVICE, non_blocking=True)
            predicted = model(batch_data)

            # Convert to either 0 or 1 based on prediction probability
            predicted = (predicted >= prob).float()
            batch_loss = loss_fn(predicted, labels.float())
            validation_loss += batch_loss.item()
            correct += (predicted == labels).type(torch.float).sum().item()

            # To compute on F1
            predictions.extend(predicted.to('cpu').numpy())
            truth.extend(labels.to('cpu').numpy())

    # To compute on F1
    pred_np_arr = np.array(predictions)
    pred_tensor = torch.tensor(pred_np_arr)
    truth_tensor = torch.tensor(truth)
    f1 = binary_f1_score(pred_tensor, truth_tensor)
    print(f"\nValidation F1 score: {f1:.5f}", flush=True)

    validation_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {validation_loss:>8f}", flush=True)

    return validation_loss

def test(dataloader: DataLoader, model: nn.Module, data: str) -> None:
    r""" Tests a trained model, reporting recall, precision, F1
    at multiple probability criteria levels

    Args:
        dataloader: Contains batches of data
        model: The model being used
        data: The data being used, either freq or dm
    """

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    truth = []
    predictions = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch_idx, (freq_data, dm_data, labels) in enumerate(dataloader):
            batch_data = None

            # Load labels to device
            labels = labels.to(DEVICE, non_blocking=True)
            
            # Load data to device and make predictions
            if data == "freq":
                batch_data = freq_data
            elif data == "dm":
                batch_data = dm_data
            else:
                print(f"Invalid data type provided: {data}")
                sys.exit(0)

            batch_data = batch_data.to(DEVICE, non_blocking=True)
            predicted = model(batch_data)

            predictions.extend(predicted.to('cpu').numpy())
            truth.extend(labels.to('cpu').numpy())

    pred_np_arr = np.array(predictions)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        binary_pred = (pred_np_arr >= threshold)
        pred_tensor = torch.tensor(binary_pred)
        truth_tensor = torch.tensor(truth)
        recall = binary_recall(pred_tensor, truth_tensor)
        precision = binary_precision(pred_tensor, truth_tensor)
        f1 = binary_f1_score(pred_tensor, truth_tensor)

        print(f"\n--- Test results: Threshold {threshold} --", flush=True)
        print(f"\tRecall: {(100*recall):.2f}%", flush=True)
        print(f"\tPrecision: {(100*precision):.2f}%", flush=True)
        print(f"\tF1: {(100*f1):.2f}%", flush=True)

def main() -> None:
    r""" Entry point for running via command line
    Transfer training for an individual pre-trained Torchvision model
    """

    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)"
    )
    parser.add_argument(
        "-g", "--gpu_id", help="GPU ID", type=int, required=False, default=0
    )
    parser.add_argument(
        "-trn",
        "--train_data_dir",
        help="Directory containing h5 file(s) for training.  Assumes the file(s) contain labels",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tst",
        "--test_data_dir",
        help="Directory containing h5 file(s) for testing.  Assumes the file(s) contain labels",
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
        help="Place to save the final best weights ",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m", "--model", help="Name of the model to train", required=True, type=str
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="Training learning rate", default=1e-3, type=float
    )
    parser.add_argument(
        "-pr", "--probability", help="Detection threshold", default=0.5, type=float
    )
    parser.add_argument(
        "-pa", "--patience", help="Num epochs with no improvement after which training will be stopped", default=3, type=int
    )
    parser.add_argument(
        "-d", "--data", help="Type of data being used. Should be freq or dm", default="freq", type=str
    )

    args = parser.parse_args()

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    print(f"Using {DEVICE} for computation", flush=True)

    if args.data not in ["freq", "dm"]:
        print(f"Invalid data type {args.data}.  Should be freq or dm")
        sys.exit(0)
    else:
        print(f"Using {args.data} data", flush=True)
    
    # Load training and split 85% to 15% into train/validate
    train_data_files = glob.glob(args.train_data_dir + "/*.h*5")
    train_data = PulsarData(files=train_data_files)
    train_data, validate_data = random_split(train_data, [0.85, 0.15])

    tr_dataloader = DataLoader(train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    v_dataloader = DataLoader(validate_data, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    best_model_path = ""
    best_vloss = float('inf')
    best_unfrozen = 0

    print(f"**** Initial training with all layers frozen ****", flush=True)
    epochs_without_improvement = 0

    # Setup model
    model = TorchvisionModel(args.model, 1, 0).to(DEVICE)

    # Setup training parameters
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    for t in range(args.epochs):
        print(f"-------------------------------", flush=True)
        print(f"Epoch {t+1}\n-------------------------------", flush=True)

        # Train the model
        print(f"Training...", flush=True)
        train_loop(tr_dataloader, model, args.data, loss_fn, optimizer, args.batch_size)

        # Validate the model and track best model perfomance
        print(f"\nPerforming validation...", flush=True)
        avg_vloss = validate_loop(v_dataloader, model, args.data, loss_fn, args.probability)
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_path = f"model_{0}_{args.model}_{args.data}_epoch{t+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"\nEpochs without improvement {epochs_without_improvement}", flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"Stopping training early", flush=True)
            break

    # Number of unfrozen layers
    n = 0 

    # Number of consecutive layers unfrozen without improvement
    consec_layers = 0 

    while consec_layers < 3:
        # Increment unfrozen count
        n += 1
        print(f"**** Training model with {n} unfrozen layers ****", flush=True)

        # Setup model
        model = TorchvisionModel(args.model, 1, n).to(DEVICE)

        # Setup training parameters
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

        # Start of training/validation
        epochs_without_improvement = 0

        for t in range(args.epochs):
            print(f"-------------------------------", flush=True)
            print(f"Epoch {t+1}\n-------------------------------", flush=True)

            # Train the model
            print(f"Training...", flush=True)
            train_loop(tr_dataloader, model, args.data, loss_fn, optimizer, args.batch_size)

            # Validate the model and track best model perfomance
            print(f"\nPerforming validation...", flush=True)
            avg_vloss = validate_loop(v_dataloader, model, args.data, loss_fn, args.probability)
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_unfrozen = n
                best_model_path = f"model_{n}_{args.model}_{args.data}_epoch{t+1}.pth"
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
                consec_layers = 0
            else:
                epochs_without_improvement += 1

            print(f"\nEpochs without improvement count {epochs_without_improvement}", flush=True)
            print(f"Value of consec_layers: {consec_layers}", flush=True)

            # As I understsand the training procedure in the paper
            # Essentially need to go 3 consecutive unfrozen layers
            # with no improvement in validation loss.
            # Specifically three consecutive layers where no improvement
            # in first 3 epochs for each layer
            if epochs_without_improvement >= args.patience:
                # Possibly increase consec layers without improvement
                if t == 2:
                    consec_layers += 1
                print(f"Stopping training early", flush=True)
                break

    print(f"\n--- TRAINING SUMMARY ---", flush=True)
    print(f"\t--- Observation counts for training data ---", flush=True)
    printObsCounts(train_data)
    print(f"\n\t--- Observation counts for validation data ---", flush=True)
    printObsCounts(validate_data)
    print(f"\n\tBest validation loss: {best_vloss}", flush=True)
    print(f"\tUnfrozen layers with best validation loss: {best_unfrozen}\n\n", flush = True)

    # Save the final best model based on train/validation to output dir
    outfile = f"{args.output_path}/{args.model}_{args.data}.pth"
    copy(best_model_path, outfile)

    # Test model
    tst_dataloader = None
    if args.test_data_dir is not None:
        model = TorchvisionModel(args.model, 1, 0)
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        model.to(DEVICE)
    
        test_data_files = glob.glob(args.test_data_dir + "/*.h*5")
        test_data = PulsarData(files=test_data_files)
        print(f"--- Observation counts for test data ---", flush=True)
        printObsCounts(test_data)
        tst_dataloader = DataLoader(test_data, batch_size=args.batch_size, pin_memory=True, shuffle=False)
        
        test(tst_dataloader, model, args.data)