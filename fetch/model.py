r""" Contains building blocks for the different pulsar models.
All the models have the same broad architecture.  The biggest
difference between them is the CNN model used to
to process the freq and dm data.
"""
import torch
import torch.nn as nn
import torchvision.models as models

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TorchvisionModel(nn.Module):
    PARAMS = {"DenseNet121": 1024,
             "DenseNet169": 1664,
             "DenseNet201": 1920,
             "VGG16": 512,
             "VGG19": 512,
             "Inception_V3": 2048,
    }   
    def __init__(self, model_name: str, out_features: int, unfreeze_layers: int = 0) -> None:
        r"""
        
        Creates a processing block containing a pre-trained torchvision
        model like DenseNet121
        
        :param model_name: The name of the pre-trained model to use
        :param out_features: Number of output features for classifier 
                             This is the k training hyperparamter
                             referred to in the original FETCH paper
        :param unfreeze_layers: Number of layers to unfreeze
        """
        super().__init__()
        
        print(f"Initializing torchvision model {model_name}", flush=True)

        self.model_name = model_name
        weights = f"{model_name}_Weights.DEFAULT"
        features = self.PARAMS[model_name]

        # Make input data compatible with pre-trained network
        self.block1= nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2, stride=(1, 1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU(),
        )

        # Get the pre-trained model from PyTorch
        self.model = torch.hub.load("pytorch/vision", model_name.lower(), weights=weights)

        # Freeze all layers to start
        for param in self.model.parameters():
            param.requires_grad = False

        if self.model_name.startswith("DenseNet"):
            self._freeze_densenet(unfreeze_layers)
        
        # Replace/set the classifier layer
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=features, out_features=out_features),
            nn.Dropout(p=0.3),
        )

    def _unfreeze_densenet(self, unfreeze_layers: int) -> None:
        """
        Go through each dense layer in each dense block and enable
        gradients until we hit the layer count or run out of layers
        """

        if unfreeze_layers == 0:
            return

        count = 0

        for layer in reversed(list(model.features.denseblock4.children())):
            count += 1
            for param in layer.parameters():
                param.requires_grad = True

            if count == unfreeze layers:
                return

        for layer in reversed(list(model.features.denseblock3.children())):
            count += 1
            for param in layer.parameters():
                param.requires_grad = True

            if count == unfreeze layers:
                return

        for layer in reversed(list(model.features.denseblock2.children())):
            count += 1
            for param in layer.parameters():
                param.requires_grad = True

            if count == unfreeze layers:
                return

        for layer in reversed(list(model.features.denseblock1.children())):
            count += 1
            for param in layer.parameters():
                param.requires_grad = True

            if count == unfreeze layers:
                return
            
    def _unfreeze_vgg(self, num_blocks: int) -> None:
        pass

    def _unfreeze_inception3(self, num_blocks: int) -> None:
        pass

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.block1(data)
        output = self.model(output)

        return output.squeeze()

class PulsarModel(nn.Module):
    def __init__(self, freq_module: nn.Module, dm_module: nn.Module, k: int) -> None:
        r"""
        
        Builds a combined pulsar prediction model using pre-trained freq and dm modules

        :param freq_module: A pre-trained nn.Module for frequency processing
        :param dm_module: A pre-trained nn.Module for dm processing
        :param k: This is the k training hyperparamter
                  referred to in the original FETCH paper
        """
        super().__init__()
    
        print(f"Building pulsar model using pre-trained modules", flush=True)

        self.freq_model = freq_module
        self.dm_model = dm_module

        # Final process of combined freq and DM data
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=k, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=k, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, freq_input: torch.Tensor, dm_input: torch.Tensor) -> torch.Tensor:
        freq_output = self.freq_model(freq_input)
        dm_output = self.dm_model(dm_input)

        # Combine the outputs and produce final classification
        output = torch.mul(freq_output, dm_output)
        output = self.classifier(output)

        return output.squeeze()