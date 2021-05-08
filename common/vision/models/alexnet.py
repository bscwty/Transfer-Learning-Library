import torch.nn as nn
from torchvision import models

__all__ = ['alexnet']

class Alexnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = models.alexnet(pretrained = True)
        self.backbone.classifier = nn.Sequential()

    def forward(self, x):
        x = self.backbone(x)
        return x

def alexnet():
    return Alexnet()