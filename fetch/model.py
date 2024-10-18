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
    def __init__(self, model_name: str, out_features: int) -> None:
        r"""
        
        Creates a processing block containing a pre-trained torchvision
        model like DenseNet121
        
        :param model_name: The name of the pre-trained model to use
        :param out_features: Number of output features for classifier 
                             This is the k training hyperparamter
                             referred to in the original FETCH paper
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
        self.pretrained = torch.hub.load("pytorch/vision", model_name.lower(), weights=weights)

        # Freeze all layers
        for layer in self.pretrained.children():
            for parameter in layer.parameters():
                parameter.requires_grad = False

        # Original setup
        # Freeze all layers of the pre-trained model at least initially
        #self.pretrained = nn.Sequential(*[i for i in list(self.pretrained.children())[:-1]])
        #for child  in self.pretrained.children():
        #    for param in child.parameters():
        #        param.requires_grad = False
        

        # Replace the classifier layer with custom sequence
        self.pretrained.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=1),
            nn.BatchNorm2d(num_features=features, eps=0.001, momentum=0.99),
            nn.Dropout(p=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=features, out_features=out_features),
            nn.Softmax(dim=1),
        )

    def freeze_block(self, num_blocks: int) -> nn.Module:
        """
        Tricky because different torchvision models are setup differently

        https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
        https://stackoverflow.com/questions/69278507/unfreeze-model-layer-by-layer-in-pytorch
        https://python.plainenglish.io/how-to-freeze-model-weights-in-pytorch-for-transfer-learning-step-by-step-tutorial-a533a58051ef
        https://stackoverflow.com/questions/62523912/freeze-certain-layers-of-an-existing-model-in-pytorch

        Suggested online
        # This is for a Graph Neural Network based on the GIN paper
        _FREEZE_KEY = {'0': ['ginlayers.0','linears_prediction_classification.0'],
               '1': ['ginlayers.1','linears_prediction_classification.1'],
               '2': ['ginlayers.2','linears_prediction_classification.2'],
               '3': ['ginlayers.3','linears_prediction_classification.3'],
               '4': ['ginlayers.4','linears_prediction_classification.4'],
               }
        def freeze_model_weights(model, freeze_key_id="0"):
   
            print('Going to apply weight frozen')
            print('before frozen, require grad parameter names:')
            for name, param in model.named_parameters():
                if param.requires_grad:print(name)
            freeze_keys = _FREEZE_KEY[freeze_key_id]
            print('freeze_keys', freeze_keys)
            for name, para in model.named_parameters():
                if para.requires_grad and any(key in name for key in freeze_keys):
                    para.requires_grad = False
            print('after frozen, require grad parameter names:')
            for name, para in model.named_parameters():
                if para.requires_grad:print(name)
            return model

        model = freeze_model_weights(model, freeze_key_id = "0")
        non_frozen_parameters = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.Adam(non_frozen_parameters, lr=0.001)
        
        """
        pass

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.block1(data)
        output = self.pretrained(output)

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
        self.block = nn.Sequential(
            nn.BatchNorm1d(num_features=k, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Linear(in_features=k, out_features=k),
            nn.Sigmoid(),
        )

    def forward(self, freq_input: torch.Tensor, dm_input: torch.Tensor) -> torch.Tensor:
        freq_output = self.freq_model(freq_input)
        dm_output = self.dm_model(dm_input)

        # Combine the outputs and produce final classification
        output = torch.mul(freq_output, dm_output)
        output = self.block(output)

        return output