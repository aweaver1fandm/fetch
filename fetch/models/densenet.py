''' Based on a DenseNet architecture
    See for example https://medium.com/@karuneshu21/implement-densenet-in-pytorch-46374ef91900 
    Modified to handle pulsar data inputs '''

import torch
import torch.nn as nn

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')