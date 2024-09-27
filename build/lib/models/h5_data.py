import logging
import os

import h5py

import torch
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
logger = logging.getLogger(__name__)

def get_data(file: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = h5py.File(file, 'r')

    freq_data = np.array(data["data_freq_time"][:])
    dm_data = np.array(data["data_dm_time"][:])
    labels = np.array(data["data_labels"][:])

    freq_tensor = torch.from_numpy(freq_data)
    dm_tensor = torch.from_numpy(dm_data)

    print(type(freq_tensor))
    print(freq_tensor.size())
    print(type(dm_tensor))
    print(dm_tensor.size())
    
    print("Before reshape")
    print(freq_data.shape)
    print(freq_data)
    new_freq_data = np.reshape(freq_data, (40000, 256, 256))

    #print("After reshape")
    #print(new_freq_data[:1])

    return freq_tensor, dm_tensor

if __name__ == "__main__":

    freq, dm = get_data("/data/fetch_data/train_data.hdf5")