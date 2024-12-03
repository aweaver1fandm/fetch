import os
import sys

import h5py

import torch
import torch.nn as nn
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
        r""" A set of pulsar observations consisting of
        1. Frequency information,
        2. DM information
        3. Label, pulsar or not (optional)
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
        self.ft_data = np.empty((0, *self.ft_dim)) # channels x dim x dim
        self.dt_data = np.empty((0, *self.dt_dim)) # channels x dim x dim
        self.labels = np.empty(0, dtype=int)
        
        for f in files:
            self._data_from_h5(f)

    def pin_memory(self):
        r""" Because it's a custom data type, need this
         function in order to pin the memory
         """
        self.ft_data = self.ft_data.pin_memory()
        self.dt_data = self.dt_data.pin_memory()
        self.labels = self.labels.pin_memory()

        return self
    
    def __len__(self)-> int:
        return self.num_observations

    def __getitem__(self, index: int)-> tuple:
        '''
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
        return torch.from_numpy(ft_data), torch.from_numpy(dt_data), torch.tensor(self.labels[index])'''

        return torch.from_numpy(self.ft_data[index]), torch.from_numpy(self.dt_data[index]), torch.tensor(self.labels[index])
        
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

        # Do a few basic data checks
        freq_data_shape = freq_data.shape
        dm_data_shape = dm_data.shape
        if freq_data_shape != dm_data_shape:
            print(f"ERROR: freq data shape({freq_data_shape}) and dm data({dm_data_shape}) shape do not match", flush=True)
            sys.exit(1)

        freq_data_size = len(freq_data.shape)
        dm_data_size = len(dm_data.shape)
        if freq_data_size != dm_data_size:
            print(f"ERROR: freq({freq_data_size}) and dm data({dm_data_size}) formats do not match")
            sys.exit(1)

        num_channels = 1
        num_obs = 0
        freq_dims = None
        dm_dims = None
        
        """ Need to handle different .h5 data size situations
        By assuming the following situations
        
        Size 4: Num observations x dim x dim x channels
        Size 3: Check 3rd value in shape
                - Equals the channel value then dim x dim x channels
                - Otherwise num observations x dim x dim
        Size 2: dim x dim

        Ultimately want num observations x channel x dim x dim
        """
        if data_size == 4:
            num_obs = data_shape[0]
            num_channels = data_shape[3]
            freq_dims = (data_shape[1], data_shape[2])
            dm_dims = (dm_data.shape[1], dm_data.shape[2])
            freq_data = np.reshape(freq_data, (self.n_channels, data_shape[1], data_shape[2]))
            dm_data = np.reshape(dm_data, (self.n_channels, data_shape[1], data_shape[2]))
        elif (data_size == 3) and (data_shape[2] == self.n_channels):
            num_obs = 1
            num_channels = data_shape[2]
            freq_dims = (data_shape[0], data_shape[1])
            dm_dims = (dm_data.shape[0], dm_data.shape[1])
            freq_data = np.reshape(freq_data, (1, data_shape[0], data_shape[1]))
            dm_data = np.reshape(dm_data, (1, data_shape[0], data_shape[1]))
        elif data_size == 3:
            num_obs = data_shape[0]
            freq_dims = (data_shape[1], data_shape[2])
            dm_dims = (dm_data.shape[1], dm_data.shape[2])
            freq_data = np.reshape(freq_data, (self.n_channels, data_shape[1], data_shape[2]))
            dm_data = np.reshape(dm_data, (self.n_channels, data_shape[1], data_shape[2]))
        elif data_size == 2:
            num_obs = 1
            freq_dims = (data_shape[0], data_shape[1])
            dm_dims = (dm_data.shape[0], dm_data.shape[1])
            freq_data = np.reshape(freq_data, (1, data_shape[0], data_shape[1]))
            dm_data = np.reshape(dm_data, (1, data_shape[0], data_shape[1]))
        else:
            print(f"ERROR: {file} contains one or more observations in an unexpected format...{data_shape}", flush=True)
            sys.exit(1)

        #  Do a few more basic data checks
        if num_channels != self.n_channels:
            print(f"Mismatch in channel information. Data has {num_channels}, expected {self.n_channels}", flush=True)
            sys.exit(1)

        if (freq_dims != self.ft_dim) or (dm_dims != self.dt_dim):
            print(f"ERROR: Data shape mismatch", flush=True)
            print(f"\tFrequency dimensions: {freq_dims}, expected {self.ft_dim}", flush=True)
            print(f"\tDM dimensions: {dm_dims}, expected {self.dt_dim}", flush=True)
            sys.exit(1)

        self.ft_data = np.append(self.ft_data, ft_data, axis=0)
        self.dt_data = np.append(self.dt_data, dt_data, axis=0)
        
        # Handle the labels if they exist
        if "data_labels" in data:
            print(f"Input file contain labels...adding to PulsarData", flush=True)
            self.labels = np.append(self.labels, data["data_labels"])
        else:
            self.labels = np.append(self.labels, np.empty(num_obs, dtype=int))

""" -- AI Code --
import torch
import scipy.signal as signal

def detrend_tensor(tensor, axis=-1, type='linear'):
    
    return torch.tensor(signal.detrend(tensor.numpy(), axis=axis, type=type))

# Example usage:
data = torch.randn(100, 5)
detrended_data = detrend_tensor(data) 

### Manual way ###
import torch

def linear_detrend(tensor, axis=-1):
    n = tensor.shape[axis]
    x = torch.arange(n).to(tensor.device) 
    x_mean = x.mean()
    y_mean = tensor.mean(dim=axis, keepdim=True)

    slope = torch.sum((x - x_mean) * (tensor - y_mean), dim=axis, keepdim=True) / torch.sum((x - x_mean) ** 2, dim=axis, keepdim=True)
    intercept = y_mean - slope * x_mean

    return tensor - slope * x - intercept

# Example usage:
data = torch.randn(100, 5)
detrended_data = linear_detrend(data) 

### Third way ###
import numpy as np
from scipy import signal

# Create a sample 3D vector
data = np.random.rand(10, 5, 3)

# Detrend along each axis
detrended_data = np.zeros_like(data)
for i in range(data.shape[2]):
    detrended_data[:, :, i] = signal.detrend(data[:, :, i])

print(detrended_data)

"""