"""
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
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
        list_IDs,
        labels,
        ft_dim=(256, 256),
        dt_dim=(256, 256),
        n_channels=1,
        n_classes=2,
        noise=False,
        noise_mean=0.0,
        noise_std=1.0,
    ):
        """

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

    def __len__(self):
        """

        :return: Number of files in data set
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
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

    def on_epoch_end(self):
        """

        :return: Updates the indices at the end of the epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)