#!/usr/bin/env python3

import argparse
import glob
import os
import string
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from fetch.pulsar_data import PulsarData
from fetch.model import PulsarModel, TorchvisionModel

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    r""" Entry point for running via command line
    Uses a pre-trained combined model to make predictions
    """

    allowed_models = TorchvisionModel.PARAMS.keys()

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
        "-m",
        "--model_dir",
        help="Base directory where models are at (or will be saved to)\n \
              Base directory will then have freq, dm, combined subdirectories (which will be created if they don't exist)\n \
              If not specified, it will default to current directory.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-fm", "--freq_model", help="Freq data processing model", type=str, default=None, choices=allowed_models
    )
    parser.add_argument(
        "-dm", "--dm_model", help="DM data processing model", type=str, default=None, choices=allowed_models
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    args = parser.parse_args()

    print(f"Using {DEVICE} for computation", flush=True)
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    # Use current directory or command line argument as base directory for model locations
    if args.model_dir is None:
        args.model_dir = os.getcwd()
    elif not os.path.isdir(args.model_dir):
        print(f"Model directory, {args.model_dir} does not exist")
        sys.exit(0)

    # Make sure at least one model has been set
    if (args.freq_model is None) and (args.dm_model is None):
        print(f"No model chosen.  At least one -fm or -dm must be specified")
        sys.exit(0)
    
    # Figure out which model we are using
    model = None
    if (args.freq_model) and (args.dm_model):
        model_name = f"{args.freq_model}_{args.dm_model}"
        model_files = os.listdir(os.path.join(args.model_dir, "combined"))
        for file in model_files:
            file_base = os.path.splitext(file)[0]
            if file_base.startswith(model_name):
                f, d, k = file_base.split("_")
                freq = TorchvisionModel(args.freq_model, int(k))
                dm = TorchvisionModel(args.dm_model, int(k))
                model = PulsarModel(freq, dm, int(k))
                model.load_state_dict(torch.load(file, weights_only=True))
                continue
    elif args.freq_model:
        model = TorchvisionModel(args.freq_model, 1)
        model_files = os.listdir(os.path.join(args.model_dir, "freq"))
        for file in model_files:
            if file.startswith(args.freq_model):
                model.load_state_dict(torch.load(file, weights_only=True))
                continue
    else:
        model = TorchvisionModel(args.dm_model, 1)
        model_files = os.listdir(os.path.join(args.model_dir, "dm"))
        for file in model_files:
            if file.startswith(args.dm_model):
                model.load_state_dict(torch.load(file, weights_only=True))
                continue

    model.eval()
    model.to(DEVICE)
    
    # Get all the candidate files
    for data_dir in args.data_dir:

        cands_to_eval = glob.glob(f"{data_dir}/*h*5")

        if len(cands_to_eval) == 0:
            print(f"No candidates to evaluate in directory: {data_dir}", flush=True)
            continue

    # Setup the candidate data
    inputs = PulsarData(files=cands_to_eval)
    dataloader = DataLoader(inputs, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    # Make predictions in batches
    predictions = []
    probs = []
    with torch.no_grad():
        for batch_idx, (freq_data, dm_data, labels) in enumerate(dataloader):

            if args.freq_model and args.dm_model:
                freq_data = freq_data.to(DEVICE, non_blocking=True)
                dm_data = dm_data.to(DEVICE, non_blocking=True)
                predicted = model(freq_data, dm_data)
            elif args.freq_model:
                freq_data = freq_data.to(DEVICE, non_blocking=True)
                predicted = model(freq_data)
            else:
                dm_data = dm_data.to(DEVICE, non_blocking=True)
                predicted = model(dm_data)

            predicted = predicted.to('cpu').numpy()
            probs.extend(predicted)
            predictions.extend(np.round(predicted >= args.probability))

    # Save the results
    print(f"Saving final results", flush=True)
    results_dict = {}
    results_dict["candidate"] = cands_to_eval
    results_dict["probability"] = probs
    results_dict["label"] = predictions

    results_file = data_dir + f"/predict_results.csv"
    pd.DataFrame(results_dict).to_csv(results_file)