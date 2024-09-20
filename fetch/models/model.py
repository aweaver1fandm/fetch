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

# Key paramters for the different models
model_params = {"a": {"freq_cnn":"densenet121", "dm_cnn":"xception", "units":256},
                "b": {"freq_cnn":"densenet121", "dm_cnn":"vgg16", "units":32},
                "c": {"freq_cnn":"densenet169", "dm_cnn":"xception", "units":112},
                "d": {"freq_cnn":"densenet201", "dm_cnn":"xception", "units":32},
                "e": {"freq_cnn":"vgg19", "dm_cnn":"xception", "units":128},
                "f": {"freq_cnn":"densenet169", "dm_cnn":"vgg16", "units":512},
                "g": {"freq_cnn":"vgg19", "dm_cnn":"vgg16", "units":128},
                "h": {"freq_cnn":"densenet201", "dm_cnn":"inceptionv2", "units":160},
                "i": {"freq_cnn":"densenet201", "dm_cnn":"vgg16", "units":32},
                "j": {"freq_cnn":"vgg19", "dm_cnn":"inceptionv2", "units":512},
                "k": {"freq_cnn":"densenet121", "dm_cnn":"inceptionv3", "units":64},
               }

class _DMBlock(nn.Module):
    def __init__(self, model: string) -> None:
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
        cnn_layer = model_params[model]["dm_cnn"]
        units = model_params[model]["units"]

        if cnn_layer != "xception":
            self.cnn = torch.hub.load("pytorch/vision", cnn_layer, weights=None)

        # Readjust the input size for the model to match our input
        first_layer = [nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),]
        first_conv_layer.extend(list(model.features))  
        cnn_layer.features= nn.Sequential(*first_conv_layer)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU(),
            cnn_layer(),    
            nn.BatchNorm2d(num_features=dm_units, eps=0.001, momentum=0.99),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=dm_units,outfeatures=units),
        )

    def forward(self, dm: Tensor) -> Tensor:
        return self.block(dm)

class _FreqBlock(nn.Module):
    def __init__(self, model: string, cnn_layer: string) -> None:
        r"""Adds a processing group to the model for the freq data.

        The freq data is processed sequentially in the following steps
        1. Conv2D (with ReLu activation)
        2. Process through a CNN (either Densenet121, Densenet169, Densenet201, VGG19)
        3. BatchNorm2D
        4. Dropout
        5. Linear

        Args:
            cnn_layer (nn.Module): The CNN network being used
        """
        super().__init__()
        cnn_layer = model_params[model]["freq_cnn"]
        units = model_params[model]["units"]

        # Readjust the input size for the model to match our input
        first_layer = [nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),]
        first_conv_layer.extend(list(model.features))  
        cnn_layer.features= nn.Sequential(*first_conv_layer)

        self.cnn = torch.hub.load("pytorch/vision", cnn_layer, weights=None)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU(),
            cnn_layer(),
            nn.BatchNorm2d(num_features=freq_units, eps=0.001, momentum=0.99),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=freq_units,outfeatures=units),
        )

    def forward(self, dm: Tensor) -> Tensor:
        return self.block(dm)

class _CombineBlock(nn.Module):
    def __init__(self, model: string) -> None:
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
        units = model_params[model]["units"]

        self.mul = nn.mul()
        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=units, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=units,outfeatures=units),
            nn.SoftMax(dim=1)
        )

    def forward(self, freq: Tensor, dm: Tensor) -> Tensor:
        output = self.mul(freq, dm)
        return self.block(output)
    
def getModel(model: str) -> nn.Module:
    pass

