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
        "-b", "--batch_size", help="Batch size for tramamaking predictions", default=64, type=int
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
        "-d", "--data", help="Type of data being used. Should be freq or dm", default="freq", type=str
    )
    args = parser.parse_args()

    if args.model not in TorchvisionModel.PARAMS:
        raise ValueError(f"Model {args.model} is not a valid model name")

    print(f"Using {DEVICE} for computation", flush=True)
    print(f"Using {args.model} for prediction based on {args.data} data", flush=True)

    # Setup the model 
    model = TorchvisionModel(args.model, 1)
    path = os.path.split(__file__)[0]
    model.load_state_dict(torch.load(f"{args.weights}/{args.model}_{args.data}.pth", weights_only=True))
    model.eval()
    model.to(DEVICE)
    
    cands_to_eval = []
    print(f"Processing input data ", flush=true)
    for data_dir in args.data_dir:

        # Get all our candidate files
        cands_to_eval += glob.glob(f"{data_dir}/*h*5")

        if len(cands_to_eval) == 0:
            print(f"No candidates to evaluate in directory: {data_dir}", flush=True)
            continue

    # Setup the candidate data
    inputs = PulsarData(files=cands_to_eval)
    dataloader = DataLoader(inputs, batch_size=args.batch_size, shuffle=False)

    print(f"Making predictions")
    # Make predictions in batches
    predictions = []
    probs = []
    with torch.no_grad():
        for freq_data, dm_data, labels in dataloader:
    
            predicted = None

            # Load labels to device
            labels = labels.to(DEVICE)

            # Load data to device and make predictions
            if data == "freq":
                freq_data = freq_data.to(DEVICE)
                predicted = model(freq_data)
            elif data == "dm":
                dm_data = dm_data.to(DEVICE)
                predicted = model(dm_data)
               
            predicted = predicted.to('cpu').numpy()
            probs.extend(predicted[:, 1])
            predictions.extend(np.round(predicted[:, 1] >= args.probability))           

    # Save the results
    print(f"Saving final results")
    results_dict = {}
    results_dict["candidate"] = cands_to_eval
    results_dict["probability"] = probs
    results_dict["label"] = predictions

    results_file = f"results_{args.model}_{args.data}.csv"
    pd.DataFrame(results_dict).to_csv(results_file)