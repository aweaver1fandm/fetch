import argparse
import os
import string
import glob

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from torcheval.metrics.functional import binary_precision, binary_recall, binary_f1_score

from fetch.pulsar_data import PulsarData
from fetch.model import PulsarModel, MODELPARAMS

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def train_loop(dataloader: DataLoader, 
               model: nn.Module, 
               loss_fn, 
               optimizer,
               batch_size: int) -> float:
    r"""
    """
    running_loss = 0.0
    last_loss = 0.0

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    for i, (freq_data, dm_data, label) in enumerate(dataloader): 
        
        # Load model to GPU/CPU
        freq_data = freq_data.to(DEVICE)
        dm_data = dm_data.to(DEVICE)
        label = label.to(DEVICE)

        # Make prediction
        pred = model(freq_data, dm_data)

        # Compute loss and gradient
        loss = loss_fn(pred, label)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Zero  gradient for next batch
        optimizer.zero_grad()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f"{i + 1} loss: {last_loss:4f}", flush=True)
            running_loss = 0.0

    return last_loss

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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_model(dataloader: DataLoader, model: nn.Module) -> None:
    r"""
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
            pred = model(freq_data, dm_data)
            _, predicted = torch.max(pred, 1)
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
    args = parser.parse_args()

    if args.model not in MODELPARAMS:
        raise ValueError(f"Model only range from a -- k.")

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    print(f"Using {DEVICE} for computation", flush=True)

    # Load training and split 85% to 15% into train/validate
    train_data_files = glob.glob(args.train_data_dir + "/*.h*5")
    train_data = PulsarData(files=train_data_files)
    train_data, validate_data = random_split(train_data, [0.85, 0.15])

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=args.batch_size, shuffle=False)

    # Add some noise to freq data to help avoid overtraining
    print(f"Adding noise to training data", flush=True)
    for freq_data, dm_data, label in train_dataloader:
        #freq_data += torch.normal(0.0, 1.0, size=freq_data.shape)
        freq_data += torch.randn(freq_data.size()) * 1 + 0

    # Build the model and push it to proper compute device
    model = PulsarModel(args.model).to(DEVICE)

    # Setup additional training parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    # Train the model/validate model
    best_model_path = ""
    best_vloss = 1000000.0

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------", flush=True)
        avg_loss = train_loop(train_dataloader, model, loss_fn, optimizer, args.batch_size)
        
        # Perform validation to see how training is going
        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, freq_data, dm_data, labels in enumerate(validation_dataloader):
                freq_data = freq_data.to(DEVICE)
                dm_data = dm_data.to(DEVICE)
                labels = labels.to(DEVICE)

                vpreds = model(freq_data, dm_data)
                vloss = loss_fn(vpreds, labels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i+1)
        print(f"Loss training: {avg_loss:4f}  Loss validation: {avg_vloss:4f}", flush=True)

        # Track best model perfomance
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"model_{args.model}_epoch{t+1}"
            best_model_path = model_path
            torch.save(model.state_dict(), model_path)

    # Load the best saved model
    model = PulsarModel(args.model)
    model.load_state_dict(torch.load(best_model_path))

    # Test the trained model
    if args.test_data_dir is not None:
        test_data_files = glob.glob(args.test_data_dir + "/*.h*5")
        test_data = PulsarData(files=test_data_files)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        test_model(test_dataloader, model)

    # Save the model weights
    weight_file = f"/model_{args.model}_weights.pth"
    torch.save(model.state_dict(), args.output_path + weight_file)
