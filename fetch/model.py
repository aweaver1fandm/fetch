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
        :param out_features: Number of output features for classifier 
                             This is the k training hyperparamter
                             referred to in the original FETCH paper
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

        ''' Suggested by AI
        # Freeze all layers in the model
        for param in vgg16.parameters():
            param.requires_grad = False

        # Unfreeze specific layers (e.g., the last convolutional block)
        for param in vgg16.features[24:].parameters():
            param.requires_grad = True

        def freeze_densenet_block(model, block_idx):
        """
            Freezes a specific block in DenseNet121.

            Args:
                model (torchvision.models.DenseNet): The DenseNet121 model.
                block_idx (int): The index of the block to freeze (0-3).
        """

        # Get the desired block
        block = model.features.denseblock[block_idx]

        # Freeze all parameters within the block
        for param in block.parameters():
            param.requires_grad = False'''

        # Original setup
        # Freeze all layers of the pre-trained model at least initially
        #self.pretrained = nn.Sequential(*[i for i in list(self.pretrained.children())[:-1]])
        #for child  in self.pretrained.children():
        #    for param in child.parameters():
        #        param.requires_grad = False
        # or could do 
        #for layer in model.children():
        #    for parameter in layer.parameters():
        #        parameter.requires_grad = True

        for param in model.parameters():
            print(f"Parameter: {param}")
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
            nn.Softmax(dim=1),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.block1(data)
        output = self.pretrained(output)

        return output.squeeze()

class PulsarModel(nn.Module):
    PARAMS = {"a": {"freq":"DenseNet121", "dm":"xception", "k":256},
              "b": {"freq":"DenseNet121", "dm":"VGG16", "k":32},
              "c": {"freq":"DenseNe169", "dm":"xception", "k":112},
              "d": {"freq":"DenseNet201", "dm":"xception", "k":32},
              "e": {"freq":"VGG19", "dm":"xception", "k":128},
              "f": {"freq":"DenseNet169", "dm":"VGG16", "k":512},
              "g": {"freq":"VGG19", "dm":"VGG16", "k":128},
              "h": {"freq":"DenseNet201", "dm":"inceptionv2", "k":160},
              "i": {"freq":"DenseNet201", "dm":"VGG16", "k":32},
              "j": {"freq":"VGG19", "dm":"inceptionv2", "k":512},
              "k": {"freq":"DenseNet121", "dm":"Inception_V3", "k":64},
    }
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
            nn.BatchNorm1d(num_features=features, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=features, out_features=features),
            nn.Sigmoid(),
        )

    def forward(self, freq_input: torch.Tensor, dm_input: torch.Tensor) -> torch.Tensor:
        freq_output = self.freq_model(freq_input)
        dm_output = self.dm_model(dm_input)

        # Combine the outputs and produce final classification
        output = torch.mul(freq_output, dm_output)
        output = self.block(output)

        return output