from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import copy
from tabulate import tabulate
from generator import Generator
from vgg19_textgen import Vgg19 
from loss import ContentLoss, TextureLoss
from itertools import permutations 
from time import time
from dataset import Custom_Dataset
import os
import numpy as np

def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512

#loader = transforms.Compose([transforms.Resize((imsize, imsize)),  transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),])  # transform it into a torch tensor

loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = Custom_Dataset('shaiv_pics', loader)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=1, shuffle=True)
test = Custom_Dataset('shaiv_pics', loader)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=1, shuffle=True)

net = Generator()
bs = 1
current_lr = 0.1
optimizer = optim.Adam(net.parameters(), lr=current_lr)
dnet = Vgg19()

style_name = 'starry_night'
x = Custom_Dataset(style_name, loader)
xloader = torch.utils.data.DataLoader(x, batch_size = 1, shuffle = True, )
for y in xloader:
    x = y
    break

dnet = dnet.to(device)
x = x.to(device)
target_texture, _ = dnet(x)
losses = []


net = net.to(device)

w = 6.5
save_dir = f'shaiv_output/{style_name}/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(f'{save_dir}/test/')
    os.mkdir(f'{save_dir}/models/')
    
for i in range(1,401):
    print(i, end = '\r')
    net.train() 
    
    for batch_idx, data in enumerate(train_loader):

        data = data.to(device)
        optimizer.zero_grad()

        
        
        start = time()
        z = [torch.rand()] 
        output = net(data, z, device)
        #print(output)
        _, target_content = dnet(data)
        
        texture, content = dnet(output)
        
        criterion_content = ContentLoss(target_content)
        

        
        content_loss = criterion_content(content) #+ criterion_texture(texture)
        texture_loss = 0
        for layer, target_layer in zip(texture, target_texture):
            criterion_texture = TextureLoss(target_layer)

            texture_loss += (criterion_texture(layer) * 10 ** w)
        
        loss = content_loss + texture_loss 
        loss.backward()
        
        optimizer.step()
        losses.append(loss.item())
        #print(loss, end = '\r')
    if i % 10 == 0:
        b = tensor2im(data[0].unsqueeze(0))
        save_image(b, f'{save_dir}/{i}_data.png')
        c = tensor2im(output[0].unsqueeze(0))
        save_image(c, f'{save_dir}/{i}_output.png')
    if i % 100 == 0:
        state = {'net': net.state_dict(),
                'epoch': i,
                'lr': current_lr}
        torch.save(state, f'{save_dir}/models/{i}_model.h5')
    print('Content Loss', content_loss.item(), 'Texture Loss', texture_loss.item())
    if i >= 1000 and i % 200 == 0:
        current_lr = current_lr * 0.7
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr 

net.eval()

with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        print(batch_idx)
        data = data.to(device)
        optimizer.zero_grad()
        
        outputs = net(data, device)
        b = tensor2im(data[0].unsqueeze(0))
        save_image(b, f'{save_dir}/test/{batch_idx}_data_test.png')
        c = tensor2im(output[0].unsqueeze(0))
        save_image(c, f'{save_dir}/test/{batch_idx}_output_test.png')