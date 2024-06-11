import torch
import torchvision

"""
def get_encoder():
    model = torchvision.models.resnext50_32x4d(True)
    model.fc = torch.nn.Linear(2048, 257)
    return model
"""

def get_encoder(pretrained=False):
    model = torchvision.models.resnet50(pretrained)
    model.fc = torch.nn.Linear(2048, 257)
    return model
