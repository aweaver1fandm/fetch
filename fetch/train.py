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
                       data_type: str,
                       tr_dataloader: DataLoader, 
                       v_dataloader: DataLoader, 
                       model_name: str) -> str:
    r"""Performs transfer training of a model for either freq or DM data

    Args:
        args: Arguments from the original code invocation
        data: Type of data, either freq or DM
        tr_dataloader: Batches of  training data
        v_dataloader: Batches of validation data
        model_name: The model being trained
    """

    # Initialize some variables
    best_model = ""
    best_vloss = float('inf')
    unfrozen = 0 
    best_unfrozen = 0

    print(f"**** Initial training with all layers frozen ****", flush=True)
    epochs_without_improvement = 0

    # Setup model
    model = TorchvisionModel(model_name, 1, 0).to(DEVICE)

    # Setup training parameters
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    # Do an initial round of transfer training
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
            best_model = f"{model_name}_{unfrozen}.pth"
            torch.save(model.state_dict(), os.join(args.model_dir, data_type, best_model))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"\nEpochs without improvement {epochs_without_improvement}", flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"Stopping training early", flush=True)
            break

    # Now unfreeze layers to fine-tune the transfer training
    consec_layers = 0 

    while consec_layers < 3:
        # Increment unfrozen count
        unfrozen += 1
        print(f"**** Training model with {unfrozen} unfrozen layers ****", flush=True)

        # Setup model
        model = TorchvisionModel(model_name, 1, unfrozen).to(DEVICE)

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
                best_unfrozen = unfrozen
                best_model = f"{model_name}_{unfrozen}.pth"
                torch.save(model.state_dict(), os.join(args.model_dir, data_type, best_model))
                epochs_without_improvement = 0
                consec_layers = 0
            else:
                epochs_without_improvement += 1

            print(f"\nEpochs without improvement count {epochs_without_improvement}", flush=True)
            print(f"Value of consec_layers: {consec_layers}", flush=True)

            # Tracking vloss across consecutive layers
            if epochs_without_improvement >= args.patience:
                # Possibly increase consec layers without improvement
                if t == 2:
                    consec_layers += 1
                print(f"Stopping training early", flush=True)
                break

    print(f"\n--- FINAL TRAINING RESULTS ---")
    print(f"\n\tBest validation loss: {best_vloss}", flush=True)
    print(f"\tUnfrozen layers with best validation loss: {best_unfrozen}\n\n", flush = True)

    return os.join(args.model_dir, data_type, best_model)

def train_combined_model(args,
                         tr_dataloader: DataLoader, 
                         v_dataloader: DataLoader, 
                         model_name: str) -> tuple[str, int]:
    r"""Performs training for models using both freq and dm data
        Assumes that the freq and dm models have already been transfer trained

    Args:
        args: Arguments from the original code invocation
        tr_dataloader: Batches of  training data
        v_dataloader: Batches of validation data
        model_name: The model being trained
    """
    
    # Figure out the freq and dm models to use
    freq_model_name, dm_model_name = model_name.split("_")
    unfrozen_freq = 0
    freq_weight_file = ""
    model_files = os.listdir(os.join(args.model_dir, "freq"))
    for file in model_files:
        if file.startswith(freq_model_name):
            freq_weight_file =  file
            base, extension = os.path.splitext(freq_weight_file)
            tmp, unfrozen_freq = base.split("-")
            freq_weight_file = os.join(args.model_dir, "freq", freq_weight_file)

    unfrozen_dm = 0
    dm_weight_file = ""
    model_files = os.listdir(os.join(args.model_dir, "dm"))
    for file in model_files:
        if file.startswith(dm_model_name):
            dm_weight_file =  file
            base, extension = os.path.splitext(dm_weight_file)
            tmp, unfrozen_dm = base.split("-")
            dm_weight_file = os.join(args.model_dir, "dm", dm_weight_file)

    # Train over different hyperparameters of k from 2^5 to 2^9
    k_hyperparameter = [2**5, 2**6, 2**7, 2**8, 2**9]

    best_model_path = ""
    best_vloss = float('inf')
    best_k = 0

    for k in k_hyperparameter:
        print(f"\nTraining run for k={k}", flush=True)
        
        # Load saved weights for freq and dm models, ignoring classifier layer
        # because we're replacing it with new layer with different num features
        freq_model = TorchvisionModel(freq_model_name, k, unfrozen_freq)
        state_dict = torch.load(freq_weight_file, weights_only=True)
        new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.classifier")}
        freq_model.load_state_dict(new_state_dict, strict=False)

        dm_model = TorchvisionModel(dm_model_name, k, unfrozen_dm)
        state_dict = torch.load(dm_weight_file, weights_only=True)
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
                best_model = f"model_{freq_model_name}_{dm_model_name}_{k}.pth"
                torch.save(model.state_dict(), os.join(args.model_dir, "combined", best_model))
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

    return os.join(args.model_dir, "combined", best_model), best_k
    
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
            else:
                freq_data = freq_data.to(DEVICE, non_blocking=True)
                dm_data = dm_data.to(DEVICE, non_blocking=True)
                predicted = model(freq_data, dm_data)

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

    allowed_models = TorchvisionModel.PARAMS.keys()

    parser = argparse.ArgumentParser(
        description="PyTorch version of Fast Extragalactic Transient Candiate Hunter (FETCH)"
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
        "-m",
        "--model_dir",
        help="Base directory where models are at (or will be saved to)\n \
              Base directory will then have freq, dm, combined subdirectories (which will be created if they don't exist)\n \
              If not specified, it will default to current directory.",
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
        "-fm", "--freq_model", help="Freq data processing model", type=str, default=None, choices=allowed_models
    )
    parser.add_argument(
        "-dm", "--dm_model", help="DM data processing model", type=str, default=None, choices=allowed_models
    )

    args = parser.parse_args()

    print(f"Using {DEVICE} for computation", flush=True)
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    # Make sure model directories exist
    if args.model_dir is None:
        args.model_dir = os.getcwd()
    elif not os.path.isdir(args.model_dir):
        print(f"Model directory, {args.model_dir} does not exist")
        sys.exit(0)
    
    try:
        os.makedir(os.path.join(args.model_dir, "freq"))
    except FileExistsError:
        pass
    try:
        os.makedir(os.path.join(args.model_dir, "dm"))
    except FileExistsError:
        pass
    try:
        os.makedir(os.path.join(args.model_dir, "combined"))
    except FileExistsError:
        pass

    # Make sure at least one model has been set
    if (args.freq_model is None) and (args.dm_model is None):
        print(f"No model chosen.  At least one -fm or -dm must be specified")
        sys.exit(0)
    
    # Figure out which data type and model(s) we are using
    data_type = None
    model_name = None
    if (args.freq_model) and (args.dm_model):
        data_type = "combined"
        model_name = f"{args.freq_model}_{args.dm_model}"
    elif (args.freq_model):
        data_type = "freq"
        model_name = args.freq_model
    else:
        data_type = "dm"
        model = args.dm_model
   
    # Read training data and split 85% to 15% into train/validate
    train_data_files = glob.glob(args.train_data_dir + "/*.h*5")
    train_data = PulsarData(files=train_data_files)
    train_data, validate_data = random_split(train_data, [0.85, 0.15])

    # Create batches of data for training and validation
    tr_dataloader = DataLoader(train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    v_dataloader = DataLoader(validate_data, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    print(f"Using {data_type} data for training", flush=True)
    print(f"\n--- TRAINING  DATA SUMMARY ---", flush=True)
    print(f"\t--- Observation counts for training data ---", flush=True)
    printObsCounts(train_data)
    print(f"\n\t--- Observation counts for validation data ---", flush=True)
    printObsCounts(validate_data)

    if data_type != "combined":
        trained_model_path = train_single_model(args, tr_dataloader, v_dataloader, model)
    else:
        trained_model_path, best_k = train_combined_model(args, tr_dataloader, v_dataloader, model)

    # Test model
    if args.test_data_dir is not None:

        test_data_files = glob.glob(args.test_data_dir + "/*.h*5")
        test_data = PulsarData(files=test_data_files)
        print(f"\t--- Observation counts for test data ---", flush=True)
        printObsCounts(test_data)
        tst_dataloader = DataLoader(test_data, batch_size=args.batch_size, pin_memory=True, shuffle=False)

        # Setup the model and run the test
        trained_model = None
        if data_type != "combined":
            trained_model = TorchvisionModel(model, 1, 0)
        else:
            freq = TorchvisionModel(args.freq_model, best_k, 0)
            dm = TorchvisionModel(args.dm_model, best_k, 0)
            trained_model = PulsarModel(freq, dm, best_k)
        
        trained_model.load_state_dict(torch.load(trained_model_path, weights_only=True))
        trained_model.to(DEVICE)
        test(tst_dataloader, model, data_type)