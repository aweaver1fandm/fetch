#!/opt/apps/miniconda3/envs/fetch/bin/python

import glob
import pandas as pd
import os
import tqdm
import logging
import argparse

logger = logging.getLogger(__name__)


def gencandcsv(
    candsfiles,
    filelist,
    snr_th=6,
    clustersize_th=2,
    dm_min=10,
    dm_max=5000,
    label=1,
    outname=None,
    chan_mask=None,
):

    if len(candsfiles) == 0:
        raise ValueError("No candidate files provided")

    if len(filelist) == 0:
        raise ValueError("No fits/fil files provided")

    filelist.sort()
    for files in filelist:
        if not os.path.isfile(files):
            raise FileNotFoundError(f"{files} not found")

    if outname is None:
        ext = filelist[0].split(".")[-1]
        if ext == "fits" or ext == "sf" or ext == "fil":
            outname = os.path.splitext(os.path.basename(filelist[0]))[0]
        else:
            raise TypeError("Can only work with list of fits file or filterbanks")

    if outname[-4:] != ".csv":
        outname = outname + ".csv"

    cands_out = pd.DataFrame(
        columns=[
            "file",
            "snr",
            "stime",
            "width",
            "dm",
            "label",
            "chan_mask_path",
            "num_files",
        ]
    )
    cands_out.to_csv(outname, mode="w", header=True, index=False)

    for file in tqdm.tqdm(candsfiles, position=0, leave=True):
        cands_out = pd.DataFrame(
            columns=[
                "file",
                "snr",
                "stime",
                "width",
                "dm",
                "label",
                "chan_mask_path",
                "num_files",
            ]
        )
        cands = pd.read_csv(
            file,
            header=None,
            comment="#",
            delim_whitespace=True,
            names=[
                "snr",
                "ssample",
                "stime",
                "width",
                "dmidx",
                "dm",
                "cluster_size",
                "startsamp",
                "endsamp",
            ],
        )

        cands_filtered = cands[
            (cands["dm"] >= dm_min)
            & (cands["dm"] <= dm_max)
            & (cands["snr"] >= snr_th)
            & (cands["cluster_size"] >= clustersize_th)
        ]

        if len(cands_filtered) == 0:
            logger.info(f"No candidate passes the threshold criterion in {file}")
        else:
            cands_out["dm"] = cands_filtered["dm"]
            cands_out["snr"] = cands_filtered["snr"]
            cands_out["width"] = cands_filtered["width"]
            cands_out["stime"] = cands_filtered["stime"]
            cands_out["file"] = os.path.abspath(filelist[0])
            cands_out["label"] = label
            cands_out["chan_mask_path"] = chan_mask
            cands_out["num_files"] = len(filelist)
            logger.debug(f"Writing candidates in {file} to {outname}")
            cands_out.to_csv(outname, mode="a", header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Your heimdall candidate csv maker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-o",
        "--fout",
        help="Output file directory for candidate csv file",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-f",
        "--fin",
        help="Input files, can be *fits or *fil or *sf",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-c", "--heim_cands", help="Heimdall cand files", nargs="+", required=True
    )
    parser.add_argument(
        "-k",
        "--channel_mask_path",
        help="Path of channel flags mask",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "-s", "--snr_th", help="SNR Threshold", required=False, type=float, default=6
    )
    parser.add_argument(
        "-dl",
        "--dm_min_th",
        help="Minimum DM allowed",
        required=False,
        type=float,
        default=10,
    )
    parser.add_argument(
        "-du",
        "--dm_max_th",
        help="Maximum DM allowed",
        required=False,
        type=float,
        default=5000,
    )
    parser.add_argument(
        "-g",
        "--clustersize_th",
        help="Minimum cluster size allowed",
        required=False,
        type=float,
        default=2,
    )
    values = parser.parse_args()

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    gencandcsv(
        values.heim_cands,
        values.fin,
        outname=values.fout,
        chan_mask=values.channel_mask_path,
        snr_th=values.snr_th,
        clustersize_th=values.clustersize_th,
        dm_min=values.dm_min_th,
        dm_max=values.dm_max_th,
    )
