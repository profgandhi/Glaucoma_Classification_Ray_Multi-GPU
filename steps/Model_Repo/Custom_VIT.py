import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from vit_pytorch.efficient import ViT
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class VIT_Custom(nn.Module):
    def __init__(self):
        super(VIT_Custom, self).__init__()
        efficient_transformer = Linformer(dim=128, seq_len=197, depth=4, heads=8, k=64)
        self.VIT = ViT(dim=128, image_size=224, patch_size=16, num_classes=2, transformer=efficient_transformer, channels=3)
        self.VIT.mlp_head = Identity()
        self.my_new_layers = nn.Sequential(nn.Linear(138, 32),
                                           nn.ReLU(),
                                           nn.Linear(32,2),
                                           nn.Sigmoid())
    
    def forward(self, x, features):
        x = self.VIT(x)
        print('HEY',x.dtype)
        x = torch.cat((x,features),axis=1)
        x = self.my_new_layers(x)
        return x
            