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
from fetch.model import PulsarModel, TorchvisionModel

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def train_single_model(args,
                       data: str,
                       tr_dataloader: DataLoader, 
                       v_dataloader: DataLoader, 
                       model_name: str) -> None:
    r"""Performs transfer training for a model using either freq or dm data

    Args:
        args: Arguments from the original code invocation
        data: Type of data, either freq or DM
        tr_dataloader: Batches of  training data
        v_dataloader: Batches of validation data
        model_name: The model being used
    """

    # Initialize some variables
    best_model = ""
    best_vloss = float('inf')
    best_unfrozen = 0

    print(f"**** Initial training with all layers frozen ****", flush=True)
    epochs_without_improvement = 0

    # Setup model
    model = TorchvisionModel(model_name, 1, 0).to(DEVICE)

    # Setup training parameters
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    for t in range(args.epochs):
        print(f"-------------------------------", flush=True)
        print(f"Epoch {t+1}\n-------------------------------", flush=True)

        # Do a training pass
        print(f"Training...", flush=True)
        train_loop(tr_dataloader, model, data, loss_fn, optimizer, args.batch_size)

        # Validate the model and track best model perfomance
        print(f"\nPerforming validation...", flush=True)
        avg_vloss = validate_loop(v_dataloader, model, data, loss_fn, args.probability)
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model = f"{model_name}_{data}_epoch{t+1}.pth"
            torch.save(model.state_dict(), os.join(args.temp_dir, best_model))
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
        model = TorchvisionModel(model_name, 1, n).to(DEVICE)

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
                best_model = f"{model_name}_{data}_{n}_epoch{t+1}.pth"
                torch.save(model.state_dict(), os.join(args.temp_dir, best_model))
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

    print(f"\n--- FINAL TRAINING RESULtS ---")
    print(f"\n\tBest validation loss: {best_vloss}", flush=True)
    print(f"\tUnfrozen layers with best validation loss: {best_unfrozen}\n\n", flush = True)

    # Save the final best model to output dir
    outfile = f"{args.output_path}/{data}/{model_name}_{best_unfrozen}.pth"
    copy(best_model_path, outfile)

def train_combined(args,
                   tr_dataloader: DataLoader, 
                   v_dataloader: DataLoader, 
                   model_name: str) -> None:
    r"""Performs training for a combined model using both freq and dm data
        Assumes that the freq and dm models have already been transfer trained

    Args:
        args: Arguments from the original code invocation
        data: Type of data, either freq or DM
        tr_dataloader: Batches of  training data
        v_dataloader: Batches of validation data
        model_name: The model being used
    """
    
    # Train over different hyperparameters of k from 2^5 to 2^9
    k_hyperparameter = [2**5, 2**6, 2**7, 2**8, 2**9]

    # Figure out the freq and dm models to use
    freq_model_name, dm_model_name = model_name.split("_")

    best_model_path = ""
    best_vloss = float('inf')
    best_k = 0

    for k in k_hyperparameter:
        print(f"\nTraining run for k={k}", flush=True)
        
        # Load saved weights for freq model, ignoring classifier layer
        # because we're replacing it with new layer with different num features
        freq_model = TorchvisionModel(freq_model_name, k, args.unfrozen_freq)
        freq_model_path = f"model_weights/{args.freq_model}_freq.pth"
        state_dict = torch.load(freq_model_path, weights_only=True)
        new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.classifier")}
        freq_model.load_state_dict(new_state_dict, strict=False)

        # Load saved weights for freq model, ignoring classifier layer
        # because we're replacing it with new layer with different num features
        dm_model = TorchvisionModel(dm_model_name, k, args.unfrozen_dm)
        dm_model_path = f"model_weights/{args.dm_model}_dm.pth"
        state_dict = torch.load(dm_model_path, weights_only=True)
        new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.classifier")}
        dm_model.load_state_dict(new_state_dict, strict=False)

        # Setup combined model
        model = PulsarModel(freq_model, dm_model, k).to(DEVICE)

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
            train_loop(tr_dataloader, model, loss_fn, optimizer, args.batch_size)

            # Validate the model and track best model perfomance
            print(f"\nPerforming validation...", flush=True)
            avg_vloss = validate_loop(v_dataloader, model, loss_fn, args.probability)
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_k = k
                model_path = f"model_{args.freq_model}_{args.dm_model}_{k}_epoch{t+1}.pth"
                best_model_path = model_path
                torch.save(model.state_dict(), model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(f"\nEpochs without improvement {epochs_without_improvement}", flush=True)
            if epochs_without_improvement >= args.patience:
                print("Stopping training early")
                break

    print(f"\n--- FINAL TRAINING RESULTS ---", flush=True)
    print(f"\n\tBest validation loss: {best_vloss}", flush=True)
    print(f"\tBest hyperparameter: {best_k}\n\n", flush = True)

    # Save the final best model based on train/validation to output dir
    outfile = f"{args.output_path}/{args.freq_model}_{args.dm_model}_{best_k}.pth"
    copy(best_model_path, outfile)
    
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
        data: The type of data being used for training freq, dm, or both
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
        # And load data on GPU
        if data == "freq" or data == "both":
            noise = torch.randn_like(freq_data) * .1
            freq_data = freq_data + noise
            freq_data = freq_data.to(DEVICE, non_blocking=True)
        elif data == "dm" or data == "both":
            dm_data = dm_data.to(DEVICE, non_blocking=True)
        else:
            print(f"Invalid data type provided: {data}", flush=True)
            sys.exit(0)

        predicted = None
        if data == "freq":
            predicted = model(freq_data)
        elif data == "dm":
            predicted = model(dm_data)
        else:
            predicted = model(freq_data, dm_data)

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
        data: The type of data being used for training freq, dm, or both
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

            # Move data to GPU and make predictions
            predicted = None
            if data == "freq":
                freq_data = freq_data.to(DEVICE, non_blocking=True)
                predicted = model(freq_data)
            elif data == "dm":
                dm_data = dm_data.to(DEVICE, non_blocking=True)
                predicted = model(dm_data)
            elif data == "both":
                freq_data = freq_data.to(DEVICE, non_blocking=True)
                dm_data = dm_data.to(DEVICE, non_blocking=True)
                predicted = model(freq_data, dm_data)
            else:
                print(f"Invalid data type provided: {data}", flush=True)
                sys.exit(0)

            # Convert to either 0 or 1 based on prediction probability
            predicted = (predicted >= prob).float()
            batch_loss = loss_fn(predicted, labels.float())
            validation_loss += batch_loss.item()
            correct += (predicted == labels).type(torch.float).sum().item()

            # Move results to CPU for further calculation
            predictions.extend(predicted.to('cpu').numpy())
            truth.extend(labels.to('cpu').numpy())

    # Compute on F1
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
        data: The data being used, freq, dm, or both
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
            
            # Load data to GPU and make predictions
            predicted = None
            if data == "freq":
                freq_data = freq_data.to(DEVICE, non_blocking=True)
                predicted = model(freq_data)
            elif data == "dm":
                dm_data = dm_data.to(DEVICE, non_blocking=True)
                predicted = model(dm_data)
            elif data == "both":
                freq_data = freq_data.to(DEVICE, non_blocking=True)
                dm_data = dm_data.to(DEVICE, non_blocking=True)
                predicted = model(freq_data, dm_data)
            else:
                print(f"Invalid data type provided: {data}", flush=True)
                sys.exit(0)

            # Move results to CPU for further calculations
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
        "--output_dir",
        help="Base directory where trained models will be saved\n \
              Models will be saved in freq, dm, combined subdirectories based on data type\n \
              If not specified, it will default to current directory.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-tmp",
        "--temp_dir",
        help="Directory to write temporary model files to during training\n \
              Files in this directory will be removed after training is complete\n \
              If not specified it will default to the current directory",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="Training learning rate", default=1e-3, type=float
    )
    parser.add_argument(
        "-pa", "--patience", help="Num epochs with no improvement after which training will be stopped", default=3, type=int
    )
    parser.add_argument(
        "-fm", "--freq_model", help="Freq data processing model", type=str, default=None
    )
    parser.add_argument(
        "-dm", "--dm_model", help="DM data processing model", type=str, default=None
    )
    parser.add_argument(
        "-uf", "--unfrozen_freq", help="Num layers to unfreeze in freq model", type=int
    )
    parser.add_argument(
        "-ud", "--unfrozen_dm", help="Num layers to unfreeze in dm model", type=int
    )

    args = parser.parse_args()

    
    print(f"Using {DEVICE} for computation", flush=True)
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    # Make sure temp and output directories exist
    if args.temp_dir is None:
        args.temp_dir = os.getcwd()
    elif not os.path.isdir(args.temp_dir):
        print(f"Temp directory {args.temp_dir} is not a directory")
        sys.exit(0)

    if args.output_dir is None:
        args.output_dir = os.getcwd()
    elif not os.path.isdir(args.output_dir):
        print(f"Output directory {args.output_dir} is not a directory")
        sys.exit(0)
    else:
        # Create the subdirectories to hold the final trained models
        # Based on the data they are trained on
        # If they already exist, no problem
        try:
            os.makedir(os.path.join(args.output_dir, "freq"))
        except FileExistsError:
            pass
        try:
            os.makedir(os.path.join(args.output_dir, "dm"))
        except FileExistsError:
            pass
        try:
            os.makedir(os.path.join(args.output_dir, "combined"))
        except FileExistsError:
            pass

    # Make sure at least one model is being trained and it's a valid model;
    if (args.freq_model is None) and (args.dm_model is None):
        print(f"No model chosen.  At least one model must be specified via -fm or -dm")
        sys.exit(0)
    elif (args.freq_model is not None) and \
         (args.freq_model not in TorchvisionModel.PARAMS):
        print(f"Invalid model chosen to process freq data {args.freq_model}")
        sys.exit(0)
    elif (args.dm_model is not None) and \
         (args.dm_model not in TorchvisionModel.PARAMS):
        print(f"Invalid model chosen to process dm data {args.dm_model}")
        sys.exit(0)

    # Figure out which data type and model(s) we are using
    data = None
    model = None
    if (args.freq_model) and (not args.dm_model):
        data = "freq"
        model = args.freq_model
    elif (not args.freq_model) and (args.dm_model):
        data = "dm"
        model = args.dm_model
    else:
        data = "both"
        model = f"{args.freq_model}_{args.dm_model}"
   
    # Read training data and split 85% to 15% into train/validate
    train_data_files = glob.glob(args.train_data_dir + "/*.h*5")
    train_data = PulsarData(files=train_data_files)
    train_data, validate_data = random_split(train_data, [0.85, 0.15])
    print(f"Using {data} data for training", flush=True)
    print(f"\n--- TRAINING  DATA SUMMARY ---", flush=True)
    print(f"\t--- Observation counts for training data ---", flush=True)
    printObsCounts(train_data)
    print(f"\n\t--- Observation counts for validation data ---", flush=True)
    printObsCounts(validate_data)

    # Create batches of data for training and validation
    tr_dataloader = DataLoader(train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    v_dataloader = DataLoader(validate_data, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    if data != "both":
        train_individual(args, tr_dataloader, v_dataloader, model)
    else:
        train_combined(args, tr_dataloader, v_dataloader, model)

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