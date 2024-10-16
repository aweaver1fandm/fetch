import argparse
import os
import string
import glob
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from torcheval.metrics.functional import binary_precision, binary_recall, binary_f1_score

from fetch.pulsar_data import PulsarData
from fetch.model import PulsarModel, PreTrainedBlock

import pandas as pd

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def train_submodel(train: DataLoader,
                   validate: DataLoader,
                   test: DataLoader,
                   component: str,
                   batch_size: int,
                   learning_rate: float,
                   epochs: int,
                   prob: float,
    ) -> nn.Module:
    r"""

    Performs training/validation/testing for a sub-component of the PulsarModel
    The sub-component will be a pre-trained model to process either frequency or DM data

    General procedure per the paper by Devansh et al.(https://arxiv.org/pdf/1902.06343)
    """

    unfrozen_layers = 0
    best_model_path = ""
    best_vloss = 1000000.0
    DATA = {"DenseNet121": "freq",
             "DenseNet169": "freq",
             "DenseNet201": "freq",
             "VGG16": "dm",
             "VGG19": "freq",
             "Inception_V3": "dm",
             "xception": "dm",
             "inceptionv2": "dm",
    }

    print(f"Training sub-component {component}", flush=True)
    print(f"Using {DATA[component]} data", flush=True)

    # Setup model
    model = PreTrainedBlock(component, out_features=1).to(DEVICE)

    # Setup training parameters
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------", flush=True)

        # Train the model
        train_loop(train, model, DATA[component], loss_fn, optimizer, batch_size)

        # Validate the model and track best model perfomance
        avg_vloss = validate_loop(validate, model, DATA[component], loss_fn, prob)
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"model_{component}_epoch{t+1}.pth"
            best_model_path = model_path
            torch.save(model.state_dict(), model_path)

    # Load the best model.  Even if not testing it we need to return it
    model = PreTrainedBlock(component, out_features=1)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    # Perform testing
    if test is not None:
        model.to(DEVICE)
        test_model(test, model, DATA[component], prob)

    return model
    
def train_fullmodel(train: DataLoader,
                   validate: DataLoader,
                   test: DataLoader,
                   component: str,
                   batch_size: int,
                   learning_rate: float,
                   epochs: int,
                   prob: float,
    ) -> nn.Module:
    r"""

    Performs training/validation for the full PulsarModel

    General procedure per the paper by Devansh et al.(https://arxiv.org/pdf/1902.06343)
    """

    best_model_path = ""
    best_vloss = 1000000.0

    # Setup model
    model = PulsarModel(component).to(DEVICE)

    # Setup training parameters
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------", flush=True)

        # Train the model
        train_loop(train, model, "all", loss_fn, optimizer, batch_size)

        # Validate the model and track best model perfomance
        avg_vloss = validate_loop(validate, model, "all", loss_fn, prob)
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"model_{component}_epoch{t+1}.pth"
            best_model_path = model_path
            torch.save(model.state_dict(), model_path)

    # Load the best model.  Even if not testing it we need to return it
    model = PulsarModel(component)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    # Perform testing
    if test is not None:
        model.to(DEVICE)
        test_model(test, model, "all", prob)

    return model

def train_loop(dataloader: DataLoader, 
               model: nn.Module,
               data: str,
               loss_fn, 
               optimizer,
               batch_size: int,
    ) -> None:
    r"""

    Perform a single pass of training for some model or model sub-component
    """

    size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    for batch, (freq_data, dm_data, label) in enumerate(dataloader):

        # Add some noise to freq data to help avoid overtraining
        #freq_data += torch.randn(freq_data.size()) * 1.0 + 0.0
        
        # Load model to GPU/CPU
        freq_data = freq_data.to(DEVICE)
        dm_data = dm_data.to(DEVICE)
        label = label.to(DEVICE)

        # Make prediction and compute loss
        if data == "all":
            pred = model(freq_data, dm_data)
        elif data == "freq":
            pred = model(freq_data)
        else:
            pred = model(dm_data)
        loss = loss_fn(pred, label.float())

        # Backpropogate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(freq_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    
def validate_loop(dataloader: DataLoader, 
                  model: nn.Module, 
                  data: str,
                  loss_fn,
                  prob: float,
    ) -> float:
    r"""
    
    Performs a single validation pass for some model or model sub-component
    """

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for freq_data, dm_data, label in dataloader:
            # Load model to GPU/CPU
            freq_data = freq_data.to(DEVICE)
            dm_data = dm_data.to(DEVICE)
            label = label.to(DEVICE)

            # Make prediction and compute loss
            if data == "all":
                pred = model(freq_data, dm_data)
            elif data == "freq":
                pred = model(freq_data)
            else:
                pred = model(dm_data)
            
            # Convert to either 0 or 1 based on prediction probability
            pred = (pred >= prob).float()
            test_loss += loss_fn(pred, label.float()).item()
            correct += (pred  == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n", flush=True)

    return test_loss

def test_model(dataloader: DataLoader, model: nn.Module, data: str, prob: float) -> None:
    r"""

    Performs testing on some fully trained model or model sub-component
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
        for freq_data, dm_data, label in dataloader:

            freq_data = freq_data.to(DEVICE)
            dm_data = dm_data.to(DEVICE)
            label = label.to(DEVICE)

            # Get predictions from model and move to CPU
            if data == "all":
                pred = model(freq_data, dm_data)
            elif data == "freq":
                pred = model(freq_data)
            else:
                pred = model(dm_data)

            #_, predicted = torch.max(pred, 1)
            predicted = (pred >= prob).float()
            predictions.extend(predicted.to('cpu').numpy())
            truth.extend(label.to('cpu').numpy())
            
    pred_tensor = torch.tensor(predictions)
    truth_tensor = torch.tensor(truth)
    recall = binary_recall(pred_tensor, truth_tensor)
    precision = binary_precision(pred_tensor, truth_tensor)
    f1 = binary_f1_score(pred_tensor, truth_tensor)

    print(f"--- Test results ---", flush=True)
    print(f"\tRecall: {(100*recall):.2f}%", flush=True)
    print(f"\tPrecision: {(100*precision):.2f}%", flush=True)
    print(f"\tF1: {(100*f1):.2f}%", flush=True)

def main():
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
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )

    args = parser.parse_args()

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    print(f"Using {DEVICE} for computation", flush=True)
    
    m = args.model
    batch_size = args.batch_size
    lr = args.learning_rate
    e = args.epochs
    p = args.probability

    # Load training and split 85% to 15% into train/validate
    print(f"Loading training data.  This may take some time...", flush=True)
    train_data_files = glob.glob(args.train_data_dir + "/*.h*5")
    train_data = PulsarData(files=train_data_files)
    train_data, validate_data = random_split(train_data, [0.85, 0.15])

    tr_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    v_dataloader = DataLoader(validate_data, batch_size=batch_size, shuffle=False)

    # Get test data if provided
    tst_dataloader = None
    if args.test_data_dir is not None:
        print(f"Loading test data.  This may take some time...")
        test_data_files = glob.glob(args.test_data_dir + "/*.h*5")
        test_data = PulsarData(files=test_data_files)
        tst_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    best_model = None
    # Perform training/validation and possibly testing
    print(f"--- Beginning training ---\n")
    if m in PreTrainedBlock.PARAMS:
        best_model = train_submodel(tr_dataloader, v_dataloader, tst_dataloader, m, batch_size, lr, e, p)
    elif m in PulsarModel.PARAMS:
        best_model = train_fullmodel(tr_dataloader, v_dataloader, tst_dataloader, m, batch_size, lr, e, p)
    else:
        print(f"Invalid model argument given {args.model}")
        sys.exit(1)
    
    # Save the final model weights
    weight_file = f"/model_{m}_weights.pth"
    torch.save(best_model.state_dict(), args.output_path + weight_file)