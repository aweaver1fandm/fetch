#!/usr/bin/env python3

import argparse
import glob
import os
import string

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from fetch.pulsar_data import PulsarData
from fetch.model import TorchvisionModel

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    r""" Entry point for running via command line
    Uses a Torchvision model that has been transfer-trained to make predictions
    """
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)",
    )
    parser.add_argument(
        "-g", "--gpu_id", help="GPU ID", type=int, required=False, default=0,
    )
    parser.add_argument(
        "-c",
        "--data_dir",
        help="Directory with candidate h5s.",
        required=True,
        type=str,
        action='append'
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size for making predictions", default=64, type=int
    )
    parser.add_argument(
        "-w", "--weights", help="Directory containing model weights", required=True
    )
    parser.add_argument(
        "-m", "--model", help="Name of the model to use", required=True
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    parser.add_argument(
        "-d", "--data", help="Type of data being analyzed. Should be freq or dm", default="freq", type=str
    )
    args = parser.parse_args()

    if args.model not in TorchvisionModel.PARAMS:
        raise ValueError(f"Model {args.model} is not a valid model name")

    print(f"Using {DEVICE} for computation", flush=True)
    print(f"Using {args.model} for prediction based on {args.data} data", flush=True)

    # Setup the model for binary classification
    model = TorchvisionModel(args.model, 1)
    path = os.path.split(__file__)[0]
    model.load_state_dict(torch.load(f"{args.weights}/{args.model}_{args.data}.pth", weights_only=True))
    model.eval()
    model.to(DEVICE)
    
    # Get candidate files
    cands_to_eval = []
    print(f"Processing input data ", flush=True)
    for data_dir in args.data_dir:

        cands_to_eval += glob.glob(f"{data_dir}/*h*5")

        if len(cands_to_eval) == 0:
            print(f"No candidates to evaluate in directory: {data_dir}", flush=True)
            continue

    # Setup the candidate data for GPU
    inputs = PulsarData(files=cands_to_eval)
    dataloader = DataLoader(inputs, batch_size=args.batch_size, pin_memory=True shuffle=False)

    # Make predictions in batches
    print(f"Making predictions...", flush=True)
    predictions = []
    probs = []
    with torch.no_grad():
        for freq_data, dm_data, labels in dataloader:
    
            predicted = None

            # Load labels to device
            labels = labels.to(DEVICE)

            # Load data to device and make predictions
            batch_data = None
            if args.data == "freq":
                batch_data = freq_data
            elif args.data == "dm":
                batch_data = dm_data

            batch_data = batch_data.to(DEVICE, non_blocking=True)
            predicted = model(batch_data)
               
            # Get the results from GPU
            predicted = predicted.to('cpu').numpy()
            probs.extend(predicted)
            predictions.extend(np.round(predicted >= args.probability))           

    # Save the final predictions
    print(f"Saving final results", flush=True)
    results_dict = {}
    results_dict["candidate"] = cands_to_eval
    results_dict["probability"] = probs
    results_dict["label"] = predictions

    results_file = f"results_{args.model}_{args.data}.csv"
    pd.DataFrame(results_dict).to_csv(results_file)