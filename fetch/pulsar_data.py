import os
import sys

import h5py

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.signal as s

import glob
from torch.utils.data import DataLoader

__all__ = [
    "printObsCounts",
    "PulsarData",
]

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def printObsCounts(dataset) -> None:
    r""" Prints summary information for a set of pulsar observations

    Args:
        dataset: 
    """
    pos_count = 0
    neg_count = 0
    for freq, dm, label in dataset:
        if label == 1:
            pos_count += 1
        else:
            neg_count += 1
        
    print(f"\tTotal observations: {len(dataset)}", flush=True)
    print(f"\tTotal pulsars: {pos_count}", flush=True)
    print(f"\tTotal non-pulsars: {neg_count}", flush=True)

class PulsarData(Dataset):
    def __init__(
        self,
        files: list,
        ft_dim: tuple = (256, 256),
        dt_dim: tuple = (256, 256),
        n_channels:int = 1,
    ) -> None:
        r""" A set of pulsar observations
        A pulsar observation consists of frequency information,
        dm information, and possibly a label 
        ``0`` for not a pulsar
        ``1`` for a pulsar

        Args:
            files: List of h5 files containing pulsar observations
            ft_dim: 2D shape of frequency data. Default: 256x256
            dt_dim: 2D shape of dm data.  Default: 256x256
            n_channels: Number of channels in data. Default: 1
        """
    
        self.ft_dim = ft_dim
        self.dt_dim = dt_dim
        self.files = files
        self.n_channels = n_channels

        self.num_observations = 0
        self.ft_data = np.empty((0, *self.ft_dim))
        self.dt_data = np.empty((0, *self.dt_dim))
        self.labels = np.empty(0, dtype=int)
        
        for f in files:
            self._data_from_h5(f)

    # Custom memory pinning method on custom type
    def pin_memory(self):
        for i in range(num_observations):
            self.ft_data[i] = self.ft_data[i].pin_memory()
            self.dt_data[i] = self.dt_data[i].pin_memory()
            self.labels[i] = self.labels[i].pin_memory()
        #self.ft_data = self.ft_data.pin_memory()
        #self.dt_data = self.dt_data.pin_memory()
        #self.labels = self.labels.pin_memory()

        return self
    
    def __len__(self)-> int:
        return self.num_observations

    def __getitem__(self, index: int)-> tuple:
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

        # Return data as PyTorch Tensor
        return torch.from_numpy(ft_data), torch.from_numpy(dt_data), torch.tensor(self.labels[index])
        
    def _data_from_h5(self, file: str) -> None:
        r""" Reads a single .h5 file 
        The file might represent one or multiple observations

        Assumes the following dataset names:
        data_dm_time
        data_freq_time
        data_labels (optional)

        Adds the observations to the arrays for the entire data set

        Args:
            file: The .h5 file containing the freq, dm, and possibly label for pulsar(s)
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
            print(f"ERROR: Data shape {data_dims} does not match expected dimensions {self.ft_dim}", flush=True)
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