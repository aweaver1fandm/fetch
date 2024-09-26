import logging
import os

import h5py

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.signal as s

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
logger = logging.getLogger(__name__)

class PulsarData(Dataset):
    def __init__(
        self,
        files: list,
        labels: list,
        ft_dim=(256, 256): tuple,
        dt_dim=(256, 256): tuple,
        n_channels=1: int,
        n_classes=2: int,
        noise=False: bool,
        noise_mean=0.0: float,
        noise_std=1.0: float,
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
        self.ft_data = np.empty((*self.ft_dim))
        self.dt_data = np.empty((*self.dt_dim))
        self.labels = []

        for f in files:
            logger.info(f"Processing file {f}")
            _data_from_h5(f)

    def __len__(self)-> int:
        """

        :return: Number of observations in data set
        """
        return self.num_observations

    def __getitem__(self, index: int)-> tuple(np.array, np.array, int):
        """

        :param index: index
        :return: Specific pulsar data(ft and dt) and label
        """
        
        ft_data = np.empty((*self.ft_data, self.n_channels))
        dt_data = np.empty((*self.dt_data, self.n_channels))

        # Do some processing before passing  observation to CNN 
        ft_data = s.detrend(np.nan_to_num(np.array(self.ft_data[index], dtype=np.float32).T))
        ft_data /= np.std(ft_data)
        ft_data -= np.median(ft_data)
        
        dt_data = np.nan_to_num(np.array(self.dt_data[index], dtype=np.float32))
        dt_datat_data /= np.std(data_dt)
        dt_data -= np.median(data_dt)

        ft_data = np.reshape(data_ft, (*self.ft_dim, self.n_channels))
        dt_data = np.reshape(data_dt, (*self.dt_dim, self.n_channels))

        """
        if self.noise:
            X += np.random.normal(
                loc=self.noise_mean, scale=self.noise_std, size=X.shape
            )"""
        
        return ft_data, dt_data, self.label[index]

    def _data_from_h5(file: str) -> None:
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
        logger.info(f"Data shape is {shape}")

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
            freq_data = np.reshape(freq_data, (shape[0], shape[1]))
            dm_data = np.reshape(dm_data, (shape[0], shape[1]))
        elif len(shape) == 3:
            num_observations = shape[0]
            data_dims = (shape[1], shape[2])
        elif len(shape) != 2:
            logger.error(f"{file} contains one or more observations in an unexpected format...{shape}")
            sys.exit(1)

        # Make sure the data dimensions are good
        if data_dims != self.ft_dim:
            logger.error(f"Data shape {data_dim} does not match expected dimensions {self.ft_dim}")
            sys.exit(1)

        self.ft_data = np.append(self.ft_data, freq_data)
        self.dt_data = np.append(self.dt_data, dm_data)
        self.num_observations += num_observations
        logger.info(f"Adding {num_observations} observations to data set")

        # Handle the labels if they exist
        if "data_labels" in data:
            self.labels = self.labels + data["data_labels"][:]
            logger.info(f"Input file contains labels")

if __name__ == "__main__":
    pass