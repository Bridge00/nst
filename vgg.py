#https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2
import torch
from torchvision.models import vgg19
import torch.nn as nn
class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained = True, progress = False).features)
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        texture = []
        for index,model in enumerate(self.features):
            x = model(x)
            if index in {1, 6, 11, 20, 29}:
                #print(model)
                texture.append(x)
            if index == 22:
                content = x
        return texture, content