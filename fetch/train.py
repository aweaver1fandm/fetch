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

def train_loop(dataloader: DataLoader, 
               model: nn.Module,
               data: str,
               loss_fn, 
               optimizer,
               batch_size: int,
    ) -> None:
    r"""
    Perform a single pass of training on a model
    """

    size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    for batch, (freq_data, dm_data, labels) in enumerate(dataloader):

        pred = None

        # Load labels to device
        labels = labels.to(DEVICE)

        # Add some noise to freq data to help avoid overtraining
        if data == "freq":
            noise = torch.randn_like(freq_data) * .1
            freq_data = freq_data + noise
            freq_data = freq_data.to(DEVICE)
            pred = model(freq_data)
        elif data == "dm":
            dm_data = dm_data.to(DEVICE)
            pred = model(dm_data)
        else:
            print(f"Invalid data type provided: {data}", flush=True)
            sys.exit(0)

        # Compute loss and backpropogate
        loss = loss_fn(pred, labels.float())
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
    
    Performs a single validation pass for a model
    """

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures 
    # that no gradients are computed during validation
    with torch.no_grad():
        for freq_data, dm_data, labels in dataloader:

            pred = None

            # Load labels to device
            labels = labels.to(DEVICE)

            # Load data to device and make predictions
            if data == "freq":
                freq_data = freq_data.to(DEVICE)
                pred = model(freq_data)
            elif data == "dm":
                dm_data = dm_data.to(DEVICE)
                pred = model(dm_data)
            else:
                print(f"Invalid data type provided: {data}")
                sys.exit(0)
            
            # Convert to either 0 or 1 based on prediction probability
            pred = (pred >= prob).float()
            validation_loss += loss_fn(pred, labels.float()).item()
            correct += (pred  == labels).type(torch.float).sum().item()

    validation_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {validation_loss:>8f} \n", flush=True)

    return validation_loss

def test(dataloader: DataLoader, model: nn.Module, data: str, prob: float) -> None:
    r"""

    Performs testing on fully trained model
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
        for freq_data, dm_data, labels in dataloader:
             
            pred = None

            # Load labels to device
            labels = labels.to(DEVICE)
            
            # Load data to device and make predictions
            if data == "freq":
                freq_data = freq_data.to(DEVICE)
                pred = model(freq_data)
            elif data == "dm":
                dm_data = dm_data.to(DEVICE)
                pred = model(dm_data)
            else:
                print(f"Invalid data type provided: {data}")
                sys.exit(0)

            #_, predicted = torch.max(pred, 1)
            predicted = (pred >= prob).float()
            predictions.extend(predicted.to('cpu').numpy())
            truth.extend(labels.to('cpu').numpy())
            
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
        "-fm", "--freq_model", help="Freq data processing model", required=True, type=str
    )
    parser.add_argument(
        "-dm", "--dm_model", help="DM data processing model", required=True, type=str
    )
    parser.add_argument(
        "-pa", "--patience", help="Num epochs with no improvement after which training will be stopped", default=3, type=int
    )
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

    # Load training and split 85% to 15% into train/validate
    print(f"Loading training/validation data.  This may take some time...", flush=True)
    train_data_files = glob.glob(args.train_data_dir + "/*.h*5")
    train_data = PulsarData(files=train_data_files)
    train_data, validate_data = random_split(train_data, [0.85, 0.15])

    tr_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    v_dataloader = DataLoader(validate_data, batch_size=args.batch_size, shuffle=False)
    
    # Train over different hyperparameters of k from 2^5 to 2^9
    k_hyperparameter = [2**5, 2**6, 2**7, 2**8, 2**9]

    best_model_path = ""
    best_vloss = float('inf')
    best_k = 0

    for k in k_hyperparameter:
        print(f"Training run for k={k}", flush=True)

        # Loading the saved models with strict=false ensures that
        # the classifier block, which was trained with outfeatures=1
        # will load even though the outfeatures will be k now
        freq_model_path = f"model_weights/{args.freq_model}_freq.pth"
        freq_model = TorchvisionModel(args.freq_model, out_features=k)
        freq_model.load_state_dict(torch.load(freq_model_path, weights_only=True, strict=False))

        dm_model_path = f"model_weights/{args.dm_model}_dm.pth"
        dm_model = TorchvisionModel(args.dm_model, out_features=k)
        dm_model.load_state_dict(torch.load(freq_model_path, weights_only=True, strict=False))

        # Setup model
        model = TorchvisionModel(args.freq_model, args.dm_model, out_features=k).to(DEVICE)

        # Setup training parameters
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

        # Start of training/validation
        epochs_without_improvement = 0

        for t in range(args.epochs):
            print(f"Epoch {t+1}\n-------------------------------", flush=True)

            # Train the model
            train_loop(tr_dataloader, model, "all", loss_fn, optimizer, args.batch_size)

            # Validate the model and track best model perfomance
            avg_vloss = validate_loop(v_dataloader, model, "all", loss_fn, args.prob)
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_k = k
                model_path = f"model_{args.freq_model}_{args.dm_model}_{k}_epoch{t+1}.pth"
                best_model_path = model_path
                torch.save(model.state_dict(), model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print("Stopping training early")
                break

    # Save the final best model based on train/validation to output dir
    outfile = f"{args.output_path}/{best_model_path}"
    torch.save(model.state_dict(), outfile)

    # Test model
    tst_dataloader = None
    if args.test_data_dir is not None:
        model = TorchvisionModel(args.model, out_features=1)
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        model.to(DEVICE)
    
        print(f"Loading test data.  This may take some time...", flush=True)
        test_data_files = glob.glob(args.test_data_dir + "/*.h*5")
        test_data = PulsarData(files=test_data_files)
        tst_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        
        test(tst_dataloader, model, "all", args.probability)