r""" Contains building blocks for the different models.
All the models have the same architecture.  The 
difference between them is the CNN model used to
to process the freq and dm data.
              
1. Process freq data (See _FreqLayer class)
2. Process dm data (See _DMLayer class)
3. Combine those results (using multiply)
4. BatchNormalization
5. 
"""

import torch
import torch.nn as nn
import torchvision.models as models
import xception as xc

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class _DMLayer(nn.Module):
    def __init__(self, cnn_layer: nn.Module) -> None:
        r"""Adds a processing group to the model for the DM data.

        The DM data is processed sequentially in the following steps
        1. Conv2D (with activation)
        2. Process through a CNN (either Xception, VGG16, InceptionV2, InceptionV3)
        3. BatchNorm2D
        4. Dropout
        5. Linear

        Args:
            cnn_layer (nn.Module): The CNN network being used
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU
        )

    def forward(self, dm: Tensor) -> Tensor:
        pass

class _FreqLayer(nn.Module):
    def __init__(self, cnn_layer: nn.Module) -> None:
        r"""Adds a processing group to the model for the freq data.

        The DM data is processed sequentially in the following steps
        1. Conv2D (with ReLu activation)
        2. Process through a CNN (either Densenet121, Densenet169, Densenet201, VGG19)
        3. BatchNorm2D
        4. Dropout
        5. Linear

        Args:
            cnn_layer (nn.Module): The CNN network being used
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU
        )

    def forward(self, dm: Tensor) -> Tensor:
        pass

class _CombineLayer(nn.Module):
    def __init__(self, cnn_layer: nn.Module) -> None:
        r"""Adds a processing group to model for the final processing of the
        data

        The outputs from the _FreqLayer and _DMLayer are combined 
        and processed in the following steps
        1. Multiply 
        2. BatchNorm2D
        3. ReLU 
        4. Linear (with Softmax activation)

        """
        super().__init__()
        self.mul = nn.mul()
        self.block = nn.Sequential(
            nn.ReLU
        )

    def forward(self, freq: Tensor, dm: Tensor) -> Tensor:
        output = self.mul(freq, dm)
        output = self.block(output)

        return output
    
def getModel(model: str) -> nn.Module:
    pass

