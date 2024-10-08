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

class PulsarData(Dataset):
    def __init__(
        self,
        files: list,
        ft_dim: tuple = (256, 256),
        dt_dim: tuple = (256, 256),
        n_channels:int = 1,
        n_classes: int = 2,
    ) -> None:
        r"""

        :param files: list of h5 files
        :param labels: list of labels (use fake labels when using predict)
        :param ft_dim: 2D shape (def (256, 256)
        :param dt_dim: 2D shape (def (256, 256)
        :param n_channels: number of channels in data (def = 1)
        :param n_classes: number of classes to classify data into (def = 2)
        """
    
        self.ft_dim = ft_dim
        self.dt_dim = dt_dim
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.num_observations = 0
        self.ft_data = np.empty((0, *self.ft_dim))
        self.dt_data = np.empty((0, *self.dt_dim))
        self.labels = np.empty(0, dtype=int)
        
        for f in files:
            self._data_from_h5(f)

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

        # Do some processing before passing observation to model 
        ft_data = s.detrend(np.nan_to_num(np.array(self.ft_data[index], dtype=np.float32).T))
        ft_data /= np.std(ft_data)
        ft_data -= np.median(ft_data)
        
        dt_data = np.nan_to_num(np.array(self.dt_data[index], dtype=np.float32))
        dt_data /= np.std(dt_data)
        dt_data -= np.median(dt_data)

        ft_data = np.reshape(ft_data, (self.n_channels, *self.ft_dim))
        dt_data = np.reshape(dt_data, (self.n_channels, *self.dt_dim))

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
        """

        data = h5py.File(file, 'r')
        if "data_freq_time" not in data:
            print(f"ERROR: {file} does not contain data with name data_freq_data", flush=True)
            sys.exit(1)
        if "data_dm_time" not in data:
            print(f"ERROR: {file} does not contain data with name data_dm_data", flush=True)
            sys.exit(1)
        freq_data = np.array(data["data_freq_time"][:])
        dm_data = np.array(data["data_dm_time"][:])

        shape = freq_data.shape

        num_observations = 1
        data_dims = (shape[0], shape[1])
        
        """ Need to handle different .h5 data situations
        Shape of length 4: Multiple observations in a file (e.g., 40000x256x256x1)
        Shape of length 3: Two possibilities
                           If last value is 1, then single observation (e.g., 256x256x1)
                           Otherwise assume it's multiple observations (e.g., 500x256x256)
         Shape of length 2: Single observation
        """
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
            print(f"ERROR: {file} contains one or more observations in an unexpected format...{shape}", flush=True)
            sys.exit(1)

        # Make sure the data dimensions are good
        if data_dims != self.ft_dim:
            print(f"ERROR: Data shape {data_dim} does not match expected dimensions {self.ft_dim}", flush=True)
            sys.exit(1)

        self.ft_data = np.append(self.ft_data, freq_data, axis=0)
        self.dt_data = np.append(self.dt_data, dm_data, axis=0)
        self.num_observations += num_observations
        
        # Handle the labels if they exist
        if "data_labels" in data:
            print(f"Input file does contain labels", flush=True)
            self.labels = np.append(self.labels, data["data_labels"])
        else:
            self.labels = np.append(self.labels, np.empty(num_observations, dtype=int))