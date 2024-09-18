''' All the models have the same broad architecture
    The biggest difference is in the 2nd layer where
    different blocks are used in the processing '''

import torch
import torch.nn as nn
import xception as xc

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dictionary containing the different model information
model_params = {}

class PulsarModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.flatten = nn.Flatten()
        self.freq_stack = nn.Sequential(
        )

        self.dm_stack = nn.Sequential(

        )

    def forward(self, freq, dm):
        freq = self.flatten(freq)
        dm = self.flatten(dm)

        freq_stack = self.freq_stack(freq)
        dm_stack = self.dm_stack(dm)

        mult_output = torch.mult(freq_stack, dm_stack)
        bn_output = nn.BatchNorm1d(num_features, eps=.001, momentum=0.99, affine=True, track_running_stats=True)
        
        hidden_output = nn.ReLU(bn_output)
        dense_output = nn.Linear(hidden_output)

        return logits

