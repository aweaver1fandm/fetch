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

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU)",
        type=int,
        required=False,
        default=0,
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
        "-w", "--weights", help="Directory containing model weights", required=True
    )
    parser.add_argument(
        "-m", "--model", help="Index of the model to use", required=True
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    args = parser.parse_args()

    if args.model not in list(string.ascii_lowercase)[:11]:
        raise ValueError(f"Model only range from a -- j.")

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    print(f"Using {DEVICE} for computation")

    # Get the model and set it to eval mode
    model = PulsarModel(args.model)
    path = os.path.split(__file__)[0]
    model = torch.load(f"{args.weight}/model_{args.model}_weights.pth", weights_only=True)
    model.to(DEVICE)
    model.eval()
    
    for data_dir in args.data_dir:

        # Get all our candidate files
        cands_to_eval = glob.glob(f"{data_dir}/*h*5")

        if len(cands_to_eval) == 0:
            print(f"No candidates to evaluate in directory: {data_dir}")
            continue

        # Setup the candidate data
        inputs = PulsarData(list_IDs=cands_to_eval)
        dataloader = DataLoader(inputs, shuffle=False)

        # Make predictions in batches
        predictions = []
        probs = []
        with torch.no_grad():
            for freq_data, dm_data, predictions in dataloader:
                freq_data = freq_data.to(DEVICE)
                dm_data - dm_data.to(DEVICE)

                preds = model(inputs)

                preds = preds.to('cpu').numpy()
                probs.extend(preds[:, 1])
                predictions.extend(np.round(preds[:, 1] >= args.probability))

        # Save the results
        results_dict = {}
        results_dict["candidate"] = cands_to_eval
        results_dict["probability"] = probs
        results_dict["label"] = predictions

        results_file = data_dir + f"/results_model_{args.model}.csv"
        pd.DataFrame(results_dict).to_csv(results_file)