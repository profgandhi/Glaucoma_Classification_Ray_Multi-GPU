import torch
from torch import nn
import torchvision.models as models


class Model:
    def __init__(self,name: str):
        self.Model = None
        if name == "VGG16":
            '''
            VGG16 Pretrained backbone model training
            '''
            from steps.Model_Repo.VGG16_pretrained import VGG16_finetune
            self.Model = VGG16_finetune()
        if name == "Custom_VIT":
            '''
            Training on custom vision transformer model
            '''
            from steps.Model_Repo.Custom_VIT import VIT_Custom
            self.Model = VIT_Custom()
            print(self.Model)
            
    def get_model(self):
        return self.Model
    

    
    
