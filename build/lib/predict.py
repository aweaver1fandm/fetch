#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import string

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from fetch.pulsar_data import PulsarData

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU)",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-n", "--nproc", help="Number of processors for training", default=4, type=int
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
        "-m", "--model", help="Index of the model to train", required=True
    )
    parser.add_argument(
        "-m", "--model", help="Index of the model to train", required=True
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    args = parser.parse_args()

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    if args.model not in list(string.ascii_lowercase)[:11]:
        raise ValueError(f"Model only range from a -- j.")

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
        torch.device = "gpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.device = "cpu"

    # Get the model and set it to eval mode
    logger.info(f"Getting model {args.model}")
    path = os.path.split(__file__)[0]
    model = torch.load(f"{path}/models/model_{args.model}.pth", weights_only=False)
    model.eval()
    
    for data_dir in args.data_dir:

        cands_to_eval = glob.glob(f"{data_dir}/*h5")

        if len(cands_to_eval) == 0:
            logger.warning(f"No candidates to evaluate in directory: {data_dir}")
            continue

        logging.debug(f"Read {len(cands_to_eval)} candidates")

        ''' Note: Wondering if possibly need to use a DataLoader to predict in batches
            Something like:
        def predict(model, test_loader):
            all_preds = []
            all_preds_raw = []
            all_labels = []

            for batch in test_loader:
                batch.x = torch.tensor(batch.x)
                batch.x = batch.x.reshape((-1, *batch.x.shape[2:]))
                batch.to(device)  
                pred = model(torch.tensor(batch.x).float(), 
                            #batch.edge_attr.float(),
                            batch.edge_index, 
                            batch.batch) 

                all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
                all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
                all_labels.append(batch.y.cpu().detach().numpy())'''

        # Create the input data
        inputs = PulsarData(
            list_IDs=cands_to_eval,
            labels=[0] * len(cands_to_eval),
            noise=False,
        )

        with torch.no_grad():
            probs = model(inputs)

        # Save the results
        probs = probs.detach().cpu().numpy()
        results_dict = {}
        results_dict["candidate"] = cands_to_eval
        results_dict["probability"] = probs[:, 1]
        results_dict["label"] = np.round(probs[:, 1] >= args.probability)

        results_file = data_dir + f"/results_{args.model}.csv"
        pd.DataFrame(results_dict).to_csv(results_file)