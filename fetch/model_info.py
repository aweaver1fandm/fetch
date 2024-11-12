import torch
import torch.nn as nn
import torchvision.models as models

from torchinfo import summary

from model import PulsarModel, TorchvisionModel

import sys

'''
model = TorchvisionModel("DenseNet121", 32, 0)
print(f"Model summary for DenseNet121")
summary(model, input_size=(1, 256, 256), batch_dim = 0)
print()
print()
'''
model = TorchvisionModel("VGG16", 32, 0)
print(model)

for name, module in reversed(list(model.named_modules())):
    print(name)
    if (name.startswith("model.features")):
        if (isinstance(module, nn.Conv2d)):
            print("Conv2D")
        if (isinstance(module, nn.ReLU)):
            print("ReLU")
'''
print(f"Model summary for VGG16")
summary(model, input_size=(1, 256, 256), batch_dim = 0)
print()
print()
'''
sys.exit(0)
"""
models = ["DenseNet121", "DenseNet169", "DenseNet201", "VGG16", "VGG19", "Inception_V3"]

for m in models:
    weights = f"{m}_Weights.DEFAULT"
    model = torch.hub.load("pytorch/vision", m.lower(), weights=weights)
    print(f"Model summary for {m}")
    summary(model, input_size=(3, 224, 244), batch_dim = 0)
    #for name, param in model.named_parameters():
    #    print(name)
    #for child in enumerate(model.children()):
        #print(type(child))
        #print(child)
    backbone = model.features
    for i in range(len(backbone)):
        print(f"i: {i} is component {backbone[i]}")
    #print(f"Backbone: {backbone}")
    #print(f"***** MODEL CHILDREN *****")
    #for child  in list(model.children()):
    #    print(f"Child: {child}")
    print()
    print()"""

conv_layers = []
m = "DenseNet121"
weights = f"{m}_Weights.DEFAULT"
model = torch.hub.load("pytorch/vision", m.lower(), weights=weights)
print(f"Model summary for {m}")
summary(model, input_size=(3, 224, 244), batch_dim = 0)
print()
print()

for feature in reversed(list(model.features.denseblock4.children())):
    print(f"Feature: {feature}")

m = "DenseNet121"
weights = f"{m}_Weights.DEFAULT"
model = torch.hub.load("pytorch/vision", m.lower(), weights=weights)
print(f"Model summary for {m}")
summary(model, input_size=(3, 224, 244), batch_dim = 0)
print()
print()

