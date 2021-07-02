#https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations 



class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        loss = nn.MSELoss() #F.mse_loss(input, self.target)
        return loss(input, self.target)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b * c * d)



class TextureLoss(nn.Module):

    def __init__(self, target_feature):
        super(TextureLoss, self).__init__()
        
        
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        
        G = gram_matrix(input)
        #print('G', G.size())
        #print('target', self.target.size())
        loss = nn.MSELoss() #F.mse_loss(input, self.target)
        #print(G.size(), self.target.size())
        return loss(G, self.target)