r""" Contains building blocks for the different models.
All the models have the same architecture.  The 
difference between them is the CNN model used to
to process the freq and dm data.
              
1. Process freq data (See _FreqLayer class)
2. Process dm data (See _DMLayer class)
3. Combine results (using multiply) and do additional processing
"""
import logging

import torch
import torch.nn as nn
import torchvision.models as models
#import xception as xc

# Key paramters for the different models
MODELPARAMS = {"a": {"freq_cnn":"densenet121", "dm_cnn":"xception", "units":256},
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

CNNPARAMS = {"densenet121":1024, "densenet169":1664, "densenet201":1920, "vgg19":512,
             "xception":2048, "vgg16":512, "inceptionv2":1536, "inceptionv3":2048}

logger = logging.getLogger(__name__)
LOGGINGFORMAT = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

class _DMBlock(nn.Module):
    def __init__(self, model: str) -> None:
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
        cnn_layer = MODELPARAMS[model]["dm_cnn"]
        units = MODELPARAMS[model]["units"]
        dm_units = CNNPARAMS[cnn_layer]

        self.cnn = None
        if cnn_layer != "xception":
            self.cnn = torch.hub.load("pytorch/vision", cnn_layer, weights=None)

        # Readjust the input size for the model to match our input
        # Not sure if this is needed
        #first_layer = [nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),]
        #first_conv_layer.extend(list(model.features))  
        #cnn_layer.features= nn.Sequential(*first_conv_layer)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU(),
            self.cnn,    
            nn.BatchNorm2d(num_features=dm_units, eps=0.001, momentum=0.99),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=dm_units,out_features=units),
        )

    def forward(self, dm: torch.Tensor) -> torch.Tensor:
        logger.debug(f"DMBlock forward function processing")
        return self.block(dm)

class _FreqBlock(nn.Module):
    def __init__(self, model: str) -> None:
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
        cnn_layer = MODELPARAMS[model]["freq_cnn"]
        units = MODELPARAMS[model]["units"]
        freq_units = CNNPARAMS[cnn_layer]

        # Readjust the input size for the model to match our input
        # Not sure if this is needed
        #first_layer = [nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),]
        #first_conv_layer.extend(list(model.features))  
        #cnn_layer.features= nn.Sequential(*first_conv_layer)

        self.cnn = torch.hub.load("pytorch/vision", cnn_layer, weights=None)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2,2), stride=(1,1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU(),
            self.cnn,
            nn.BatchNorm2d(num_features=freq_units, eps=0.001, momentum=0.99),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=freq_units,out_features=units),
        )

    def forward(self, dm: torch.Tensor) -> torch.Tensor:
        logger.debug(f"FreqBlock forward function processing")
        return self.block(dm)

class CompleteModel(nn.Module):
    def __init__(self, model: str) -> None:
        r"""Builds the complete model, combining the freq data sub-model
        with the dm data sub-model and final processing of the combined
        data is done
        """
        super().__init__()

        logging.basicConfig(level=logging.DEBUG, format=LOGGINGFORMAT)
        logger.debug(f"Building model {model}")
        logger.debug(f"Freq CNN is {MODELPARAMS[model]['freq_cnn']}")
        logger.debug(f"DM CNN is {MODELPARAMS[model]['dm_cnn']}")

        # https://discuss.pytorch.org/t/how-to-assemble-two-models-into-one-big-model/157027
        self.freq_model = _FreqBlock(model)
        self.dm_model = _DMBlock(model)

        units = MODELPARAMS[model]["units"]

        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=units, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=units, out_features=units),
            nn.Softmax(dim=1)
        )

    def forward(self, freq_input: torch.Tensor, dm_input: torch.Tensor) -> torch.Tensor:
        logger.debug(f"CompleteModel forward function processing")
        freq_output = self.freq_model(freq_input)
        dm_output = self.dm_model(dm_input)

        output = torch.mul(freq_output, dm_output)
        return self.block(output)
    
def getModel(model: str) -> nn.Module:
    pass

if __name__ == "__main__":
     for model in MODELPARAMS:
        print(f"Trying to build model {model}")
        m = CompleteModel(model)
