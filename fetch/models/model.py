''' All the models have the same broad architecture
    The biggest difference is in the 2nd layer where
    different blocks are used in the processing '''

import torch
import torch.nn as nn
import torchvision.models as models
import xception as xc

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dictionary containing the different model information
model_params = {}

class PulsarModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        densenet121 = models.densenet121()

        self.freq_stack = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True)
            nn.ReLU()
            densenet121
            nn.BatchNorm2d(num_features=1024, eps=0.001, mommentum=0.99)
            nn.Dropout(p=0.3)
            nn.Linear(in_features=1024, out_features=256)
        )

        self.dm_stack = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True)
            nn.ReLU()

        )
        self.bn = nn.BatchNorm2d(num_features=256, eps=.001, momentum=0.99)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=256, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, freq, dm):
        freq_stack_output = self.freq_stack(freq)
        dm_stack_output = self.dm_stack(dm)

        output = torch.mult(freq_stack_output, dm_stack_output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.linear(output)
        output = self.softmax(output)

        return output

