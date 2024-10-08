r""" Contains building blocks for the different pulsar models.
All the models have the same broad architecture.  The biggest
difference between them is the CNN model used to
to process the freq and dm data.
"""
import torch
import torch.nn as nn
import torchvision.models as models
#import xception as xc

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Key paramters for the different pulsar models
MODELPARAMS = {"a": {"freq":"densenet121", "dm":"xception", "features":256},
               "b": {"freq":"densenet121", "dm":"vgg16", "features":32},
               "c": {"freq":"densenet169", "dm":"xception", "features":112},
               "d": {"freq":"densenet201", "dm":"xception", "features":32},
               "e": {"freq":"vgg19", "dm":"xception", "features":128},
               "f": {"freq":"densenet169", "dm":"vgg16", "features":512},
               "g": {"freq":"vgg19", "dm":"vgg16", "features":128},
               "h": {"freq":"densenet201", "dm":"inceptionv2", "features":160},
               "i": {"freq":"densenet201", "dm":"vgg16", "features":32},
               "j": {"freq":"vgg19", "dm":"inceptionv2", "features":512},
               "k": {"freq":"densenet121", "dm":"inception_v3", "features":64},
}

# Paramaters related to different pre-trained CNN models
CNNPARAMS = {"densenet121": {"features":1024, "weights":"DenseNet121_Weights.DEFAULT"},
             "densenet169": {"features":1664, "weights":"DenseNet169_Weights.DEFAULT"},
             "densenet201": {"features":1920, "weights":"DenseNet201_Weights.DEFAULT"},
             "vgg19": {"features":512, "weights":"VGG19_Weights.DEFAULT"},
             "xception": {"features":2048, "weights":None},
             "vgg16": {"features":512, "weights":"VGG16_Weights.DEFAULT"},
             "inceptionv2": {"features":1536, "weights":None},
             "inception_v3": {"features":2048, "weights":"Inception_V3_Weights.DEFAULT"},
}   

class _CNNBlock(nn.Module):
    def __init__(self, model: str, data_type: str) -> None:
        r"""Adds a processing group to process either freq or DM data
        
        The  data is processed in the following steps
        1. Conv2D (with activation)
        2. Pre-trained CNN (e.g., DenseNet121)
        3. BatchNorm2D
        4. Dropout
        5. Linear

        The output of the processing of this block for both freq and DM
        data are then combined and further processed to produce a final 
        classification

        :param model: The pulsar model being built
        :param data_type: The type of data being processed, freq or dm
        """
        super().__init__()
        cnn_model = MODELPARAMS[model][data_type]
        cnn_features = MODELPARAMS[model]["features"]
        data_features = CNNPARAMS[cnn_model]["features"]

        self.block1= nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2, stride=(1, 1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU(),
            )

        self.cnn_block = None
        if cnn_model != "xception":
            self.cnn_block = torch.hub.load("pytorch/vision", cnn_model, weights=CNNPARAMS[cnn_model]["weights"])

        # Don't train any of the layers of the pre-trained models
        self.cnn_block = nn.Sequential(*[i for i in list(self.cnn_block.children())[:-1]])
        for ch in self.cnn_block.children():
            for param in ch.parameters():
                param.requires_grad = False

        # Replace the classifier layer with custom sequence
        self.cnn_block.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=1),
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(num_features=data_features, eps=0.001, momentum=0.99),
            nn.Dropout(p=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=data_features, out_features=cnn_features),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.block1(data)
        output = self.cnn_block(output)
        output = self.block2(output)
        
        return output

class PulsarModel(nn.Module):
    def __init__(self, model: str) -> None:
        r"""Builds the complete pulsar model containing three components
        
        1. Sub-model to process freq data
        2. Sub-model to process DM data
        3. Combine results from two sub-models and further
           process to produce final classification

        :param model: The pulsar model being built (a-k)
        """
        super().__init__()

        features = MODELPARAMS[model]["features"]
    
        print(f"Building pulsar model: {model}", flush=True)
        print(f"Using {MODELPARAMS[model]['freq']} for frequency data processing", flush=True)
        print(f"Using {MODELPARAMS[model]['dm']} for DM data processing", flush=True)

        self.freq_model = _CNNBlock(model, "freq")
        self.dm_model = _CNNBlock(model, "dm")

        # Final process of combined freq and DM data
        self.block = nn.Sequential(
            nn.BatchNorm1d(num_features=features, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=features, out_features=features),
            nn.Softmax(dim=1)
        )

    def forward(self, freq_input: torch.Tensor, dm_input: torch.Tensor) -> torch.Tensor:
        freq_output = self.freq_model(freq_input)
        dm_output = self.dm_model(dm_input)

        # Combine the outputs and produce final classification
        output = torch.mul(freq_output, dm_output)
        output = self.block(output)

        return output