import torch
from torch import nn
import torchvision.models as models

class VGG16_finetune(nn.Module):
    def __init__(self):
        super(VGG16_finetune, self).__init__()
        self.pretrained = models.vgg16(pretrained=True)
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
                                           nn.ReLU(),
                                           nn.Linear(100, 2),
                                           nn.Sigmoid())
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        return x
    
    

    
    
