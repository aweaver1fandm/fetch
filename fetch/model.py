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

class PreTrainedBlock(nn.Module):
    PARAMS = {"DenseNet121": 1024,
             "DenseNet169": 1664,
             "DenseNet201": 1920,
             "VGG16": 512,
             "VGG19": 512,
             "Inception_V3": 2048,
             "xception": 2048,
             "inceptionv2": 1536,
}   
    def __init__(self, model: str, 
                out_features: int, 
                unfreeze: int = 0) -> None:
        r"""
        
        Creates a processing block containing a pre-trained CNN like DenseNet
        
        :param model: The name of the pre-trained model to use
        :param out_features: Number of output features for classifier layer
        :param unfreeze: Number of layers to unfreeze for fine-tuning the network
                         Default is 0
        """
        super().__init__()
        weights = f"{model}_Weights.DEFAULT"
        features = self.PARAMS[model]

        # Make input data compatible with pre-trained network
        self.block1= nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2, stride=(1, 1), padding="valid", dilation=(1,1), bias=True),
            nn.ReLU(),
            )

        # Get the pre-trained model from PyTorch
        self.pretrained = torch.hub.load("pytorch/vision", model.lower(), weights=weights)

        # Freeze all layers of the pre-trained model
        self.pretrained = nn.Sequential(*[i for i in list(self.pretrained.children())[:-1]])
        for child  in self.pretrained.children():
            for param in child.parameters():
                param.requires_grad = False

        # Possibly unfreeze some layers
        # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
        # https://stackoverflow.com/questions/69278507/unfreeze-model-layer-by-layer-in-pytorch
        # https://python.plainenglish.io/how-to-freeze-model-weights-in-pytorch-for-transfer-learning-step-by-step-tutorial-a533a58051ef
        # https://stackoverflow.com/questions/62523912/freeze-certain-layers-of-an-existing-model-in-pytorch

        # Replace the classifier layer with custom sequence
        self.pretrained.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=1),
            nn.BatchNorm2d(num_features=features, eps=0.001, momentum=0.99),
            nn.Dropout(p=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=features, out_features=out_features),
            nn.Sigmoid(),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.block1(data)
        output = self.pretrained(output)

        return output.squeeze()

class PulsarModel(nn.Module):
    PARAMS = {"a": {"freq":"DenseNet121", "dm":"xception", "features":256},
              "b": {"freq":"DenseNet121", "dm":"VGG16", "features":32},
              "c": {"freq":"DenseNe169", "dm":"xception", "features":112},
              "d": {"freq":"DenseNet201", "dm":"xception", "features":32},
              "e": {"freq":"VGG19", "dm":"xception", "features":128},
              "f": {"freq":"DenseNet169", "dm":"VGG16", "features":512},
              "g": {"freq":"VGG19", "dm":"VGG16", "features":128},
              "h": {"freq":"DenseNet201", "dm":"inceptionv2", "features":160},
              "i": {"freq":"DenseNet201", "dm":"VGG16", "features":32},
              "j": {"freq":"VGG19", "dm":"inceptionv2", "features":512},
              "k": {"freq":"DenseNet121", "dm":"Inception_V3", "features":64},
}
    def __init__(self, model: str) -> None:
        r"""
        
        Builds a combined untrained pulsar model containing three components
        
        1. Sub-model to process freq data
        2. Sub-model to process DM data
        3. Combine results from two sub-models and further
           process to produce final classification

        :param model: The pulsar model being built (a-k)
        """
        super().__init__()

        f_model = self.PARAMS[model]['freq']
        d_model = self.PARAMS[model]['dm']
        features = self.PARAMS[model]["features"]
    
        print(f"Building full pulsar model: {model}", flush=True)
        print(f"Using {f_model} for frequency data processing", flush=True)
        print(f"Using {d_model} for DM data processing", flush=True)

        self.freq_model = PreTrainedBlock(f_model, out_features=1)
        self.dm_model = PreTrainedBlock(d_model, out_features=1)

        # Final process of combined freq and DM data
        self.block = nn.Sequential(
            nn.BatchNorm1d(num_features=features, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=features, out_features=features),
            nn.Sigmoid(),
        )

    def __init__(self, freq_module: nn.Module, dm_module: nn.Module, features: int) -> None:
        r"""
        
        Builds a combined pulsar model using pre-trained freq and dm modules

        :param freq_module: A pre-trained nn.Module for frequency processing
        :param dm_module: A pre-trained nn.Module for dm processing
        :param features: Number of features for combined processing
        """
        super().__init__()
    
        print(f"Building pulsar model using pre-trained modules", flush=True)

        self.freq_model = freq_module
        self.dm_model = dm_module

        # Final process of combined freq and DM data
        self.block = nn.Sequential(
            #nn.BatchNorm1d(num_features=features, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=features, out_features=features),
            nn.Sigmoid(),
        )

    def forward(self, freq_input: torch.Tensor, dm_input: torch.Tensor) -> torch.Tensor:
        freq_output = self.freq_model(freq_input)
        dm_output = self.dm_model(dm_input)

        print(f"Freq output shape: {freq_output.shape()}", flush=True)
        print(f"DM output shape: {dm_output.shape()}", flush=True)

        # Combine the outputs and produce final classification
        output = torch.mul(freq_output, dm_output)
        print(f"multiplied output shape: {output.shape()}", flush=True)
        output = self.block(output)

        return output