''' Based on a DenseNet architecture
    See for example https://medium.com/@karuneshu21/implement-densenet-in-pytorch-46374ef91900 
    Modified to handle pulsar data inputs '''

import torch
import torch.nn as nn

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the different model architecture parameters
model_parameters={}
model_parameters['densenet121'] = [6,12,24,16]
model_parameters['densenet169'] = [6,12,32,32]
model_parameters['densenet201'] = [6,12,48,32]

# growth rate
k = 32
compression_factor = 0.5