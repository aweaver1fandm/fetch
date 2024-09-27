import logging
import os
import sys

import h5py

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.signal as s

import glob
from torch.utils.data import DataLoader

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
logger = logging.getLogger(__name__)
LOGGINGFORMAT = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

class PulsarData(Dataset):
    def __init__(
        self,
        files: list,
        labels: list = [],
        ft_dim: tuple = (256, 256),
        dt_dim: tuple = (256, 256),
        n_channels:int = 1,
        n_classes: int = 2,
        noise:bool = False,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
    ) -> None:
        r"""

        :param files: list of h5 files
        :type files: list
        :param labels: list of labels (use fake labels when using predict)
        :type labels: list
        :param ft_dim: 2D shape (def (256, 256)
        :type dt_dim tuple
        :param dt_dim: 2D shape (def (256, 256)
        :type ft_dim tuple
        :param n_channels: number of channels in data (def = 1)
        :type n_channels: int
        :param n_classes: number of classes to classify data into (def = 2)
        :type n_classes: ints
        :param noise: to add noise or not to?
        :type noise: bool
        :param noise_mean: mean of gaussian noise
        :type noise_mean: float
        :param noise_std: standard deviation of gaussian noise
        :type noise_std: float
        """
    
        self.ft_dim = ft_dim
        self.dt_dim = dt_dim
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.noise = noise
        self.labels = labels
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.num_observations = 0
        self.ft_data = np.empty((0, *self.ft_dim))
        self.dt_data = np.empty((0, *self.dt_dim))
        self.labels = np.empty(0, dtype=int)
        
        logger.debug(f"ft data shape {self.ft_data.shape}")
        logger.debug(f"dt data shape {self.dt_data.shape}")

        for f in files:
            logger.debug(f"Processing file {f}")
            self._data_from_h5(f)

        logger.debug(f"After processing files ft data shape {self.ft_data.shape}")
        logger.debug(f"After processing files dt data shape {self.dt_data.shape}")

    def __len__(self)-> int:
        """

        :return: Number of observations in data set
        """
        return self.num_observations

    def __getitem__(self, index: int)-> tuple:
        """

        :param index: index
        :return: Specific pulsar data(ft and dt) and label
        """
        
        ft_data = np.empty((*self.ft_dim, self.n_channels))
        dt_data = np.empty((*self.dt_dim, self.n_channels))

        # Do some processing before passing observation to CNN 
        ft_data = s.detrend(np.nan_to_num(np.array(self.ft_data[index], dtype=np.float32).T))
        ft_data /= np.std(ft_data)
        ft_data -= np.median(ft_data)
        
        dt_data = np.nan_to_num(np.array(self.dt_data[index], dtype=np.float32))
        dt_data /= np.std(dt_data)
        dt_data -= np.median(dt_data)

        ft_data = np.reshape(ft_data, (self.n_channels, *self.ft_dim))
        dt_data = np.reshape(dt_data, (self.n_channels, *self.dt_dim))

        """
        if self.noise:
            X += np.random.normal(
                loc=self.noise_mean, scale=self.noise_std, size=X.shape
            )"""
        logger.debug(f"Freq data shape {ft_data.shape}")
        logger.debug(f"DM data shape {dt_data.shape}")

        return ft_data, dt_data, self.labels[index]

    def _data_from_h5(self, file: str) -> None:
        r"""

        Reads a single .h5 file 
        The file might represent one or multiple observations

        Assumes the following dataset names:
        data_dm_time
        data_freq_time
        data_labels (optional)

        Adds the observations to the arrays for the entire data set

        :param file: The .h5 file containing the freq, dm, and possibly label for pulsar(s)
        :return: None
        """

        data = h5py.File(file, 'r')
        if "data_freq_time" not in data:
            logger.error(f"{file} does not contain data with name data_freq_data")
            sys.exit(1)
        if "data_dm_time" not in data:
            logger.error(f"{file} does not contain data with name data_dm_data")
            sys.exit(1)
        freq_data = np.array(data["data_freq_time"][:])
        dm_data = np.array(data["data_dm_time"][:])

        shape = freq_data.shape
        logger.debug(f"Original data shape is {shape}")

        num_observations = 1
        data_dims = (shape[0], shape[1])
        # Need to handle different .h5 data situations
        # Shape of length 4: Multiple observations in a file (e.g., 40000x256x256x1)
        # Shape of length 3: Two possibilities
        #                    If last value is 1, then single observation (e.g., 256x256x1)
        #                    Otherwise assume it's multiple observations (e.g., 500x256x256)
        # Shape of length 2: Single observation
        if len(shape) == 4:
            freq_data = np.reshape(freq_data, (shape[0], shape[1], shape[2]))
            dm_data = np.reshape(dm_data, (shape[0], shape[1], shape[2]))
            num_observations = shape[0]
            data_dims = (shape[1], shape[2])
        elif ((len(shape) == 3) and (shape[2] == 1)):
            freq_data = np.reshape(freq_data, (1, shape[0], shape[1]))
            dm_data = np.reshape(dm_data, (1, shape[0], shape[1]))
        elif len(shape) == 3:
            num_observations = shape[0]
            data_dims = (shape[1], shape[2])
        elif len(shape) == 2:
            freq_data = np.reshape(freq_data, (1, shape[0], shape[1]))
            dm_data = np.reshape(dm_data, (1, shape[0], shape[1]))
        else:
            logger.error(f"{file} contains one or more observations in an unexpected format...{shape}")
            sys.exit(1)

        logger.debug(f"Reshaped data shape is {freq_data.shape}")

        # Make sure the data dimensions are good
        if data_dims != self.ft_dim:
            logger.error(f"Data shape {data_dim} does not match expected dimensions {self.ft_dim}")
            sys.exit(1)

        logger.debug(f"Adding {num_observations} observations to data set")
        self.ft_data = np.append(self.ft_data, freq_data, axis=0)
        self.dt_data = np.append(self.dt_data, dm_data, axis=0)
        self.num_observations += num_observations
        
        # Handle the labels if they exist
        if "data_labels" in data:
            logger.debug(f"Input file contains labels")
            self.labels = np.append(self.labels, data["data_labels"])
        else:
            logger.debug(f"Input file does not contains labels")
            self.labels = np.append(self.labels, np.empty(num_observations, dtype=int))

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, format=LOGGINGFORMAT)
    
    sample_files = glob.glob("/data/fetch_data/sample_data/*.h5")
    data_test = PulsarData(files=sample_files)

    i = 0
    for obs in data_test:
        print(f"FT data is {obs[0]}")
        print(f"DT data is {obs[1]}")
        print(f"Label data is {obs[2]}")
        i += 1

        if i == 3:
            sys.exit(0)