import logging
import os

import h5py

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.signal as s

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
logger = logging.getLogger(__name__)

class PulsarDMData(Dataset):
    def __init__(
        self,
        list_IDs: list,
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

        :param list_IDs: list of h5 files
        :type list_IDs: list
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
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()
        self.noise = noise
        self.labels = labels
        self.noise_mean = noise_mean
        self.noise_std = noise_std

class PulsarFreqData(Dataset):
    def __init__(
        self,
        list_IDs: list,
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

        :param list_IDs: list of h5 files
        :type list_IDs: list
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
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()
        self.noise = noise
        self.labels = labels
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __len__(self)-> int:
        """

        :return: Number of files in data set
        """
        return len(self.list_IDs)

    def __getitem__(self, index: int)-> tuple(np.array, np.array):
        """

        :param index: index
        :return: Specific pulsar sample and labels for given index
        """

        ft_dim = np.empty((*self.ft_dim, self.n_channels))
        dt_dim = np.empty((*self.dt_dim, self.n_channels))

        with h5py.File(index, "r") as f;
            data_ft = s.detrend(
                np.nan_to_num(np.array(f["data_freq_time"], dtype=np.float32).T)
            )
            data_ft /= np.std(data_ft)
            data_ft -= np.median(data_ft)
            data_dt = np.nan_to_num(
                np.array(f["data_dm_time"], dtype=np.float32)
            )
            data_dt /= np.std(data_dt)
            data_dt -= np.median(data_dt)

            ft_dim = np.reshape(data_ft, (*self.ft_dim, self.n_channels))
            dt_dim = np.reshape(data_dt, (*self.dt_dim, self.n_channels))

        label = self.labels[index]
        return X, label

def _data_from_dir(dir: str) -> tuple(np.array, np.array, np.array):
    r"""
    Processes all .h5 files in a directory

    :param dir: The directory containing candidate .h5 files
    :return: Three separate numpy arrays, for freq data
    dm data, and the labels for each file
    """
    return freq, dm, labels

def _data_from_h5(file: str) -> tuple(np.array, np.array, np.array):
    r"""

    Reads a single .h5 file 
    The file might represent a single observation or multiple observations

    Assumes the following dataset names:
    data_dm_time
    data_freq_time
    data_labels (optional)

    :param file: The .h5 file containing the freq, dm, and possibly label for pulsar(s)
    :return: Three separate numpy arrays, for freq data
    dm data, and the label
    """

    freq_tensor = torch.from_numpy(freq_data)
    dm_tensor = torch.from_numpy(dm_data)

    print(type(freq_tensor))
    print(freq_tensor.size())
    print(type(dm_tensor))
    print(dm_tensor.size())
    
    print(freq_data.shape)

    # Options
    # data size 2 --> single instance
    # data size 3 --> depends if last value is 1 then single instance, otherwise first value probably means multiple instances
    # data size 4 --> multiple instances
    data = h5py.File(file, 'r')
    freq_data = np.array(data["data_freq_time"][:])
    dm_data = np.array(data["data_dm_time"][:])
    labels = np.array(data["data_labels"][:])

    shape = freq_data.shape
    if len(shape)  == 4:
        freq_data = np.reshape(freq_data, (shape[0], shape[1], shape[2]))
        dm_data = np.reshape(dm_data, (shape[0], shape[1], shape[2]))
    elif ((len(freq_data) == 3) and (shape[2] == 1)):
        freq_data = np.reshape(freq_data, (shape[0], shape[1]))
        dm_data = np.reshape(dm_data, (shape[0], shape[1]))

    with h5py.File(index, "r") as f;
            data_ft = s.detrend(
                np.nan_to_num(np.array(f["data_freq_time"], dtype=np.float32).T)
            )
            data_ft /= np.std(data_ft)
            data_ft -= np.median(data_ft)
            data_dt = np.nan_to_num(
                np.array(f["data_dm_time"], dtype=np.float32)
            )
            data_dt /= np.std(data_dt)
            data_dt -= np.median(data_dt)

            ft_dim = np.reshape(data_ft, (*self.ft_dim, self.n_channels))
            dt_dim = np.reshape(data_dt, (*self.dt_dim, self.n_channels))
    return freq, dm, labels

if __name__ == "__main__":