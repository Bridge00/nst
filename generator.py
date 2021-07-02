import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from PIL import Image

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, k):
        super(Block, self).__init__()
     
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = k, stride = 1, padding = k == 3, padding_mode = 'reflect')
        self.batch =  nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)
        
        
    def forward(self, input):
        return self.relu(self.batch(self.conv(input)))

    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init_constant(m.bias, 0)
            elif isinstnace(m, nn.BatchNorm2d):
                nn.init_constant(m.weight, 1)
                nn.init_constant(m.bias, 0)
    
class Convblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convblock, self).__init__()
        

        self.block1 = Block(in_channel, out_channel, 3)
        self.block2 = Block(out_channel, out_channel, 3)
        self.block3 = Block(out_channel, out_channel, 1)
        

    def forward(self, input):
  
        out = self.block1(input)
      
        out = self.block2(out)

        out = self.block3(out)
 
        return out #self.block3(self.block2(self.block1(input)))
        


class Join(nn.Module):
    def __init__(self, in_channel): 
        super(Join, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.batch1 = nn.BatchNorm2d(in_channel)
        self.batch2 = nn.BatchNorm2d(8)
        
    def forward(self, im_lower, im_higher):

        norm_im_upsampled = self.batch1(self.up(im_lower))
        norm_im_higher = self.batch2(im_higher)
      
        
        return torch.cat((norm_im_upsampled, norm_im_higher), 1)
        
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
   
        input_channel = 6
        #self.convblocks = [Convblock(input_channel, 8) for i in range(k_num)]
        #print(self.convblocks)
        
        self.convblock1 = Convblock(input_channel, 8)
        self.convblock2 = Convblock(input_channel, 8)
        self.convblock3 = Convblock(input_channel, 8)
        self.convblock4 = Convblock(input_channel, 8)
        self.convblock5 = Convblock(input_channel, 8)
        self.convblock6 = Convblock(input_channel, 8)
        
        self.j56 = Join(8)
        self.j45 = Join(16)
        self.j34 = Join(24)
        self.j23 = Join(32)
        self.j12 = Join(40)
        
    
        #self.batches = [nn.BatchNorm2d(8 * (i)) for i in range(1, k_num)]
        #self.convblocks_after_join = [Convblock(8 * (i+1), 8 * (i+1)) for i in range(1, k_num)]
        
        self.joined_cb1 = Convblock(16, 16)
        self.joined_cb2 = Convblock(24, 24)
        self.joined_cb3 = Convblock(32, 32)
        self.joined_cb4 = Convblock(40, 40)
        self.joined_cb5 = Convblock(48, 48)
        

        self.end_block = Block(8 * (6), 3, 1)
        
        
        #self.down = nn.Upsample(scale_factor=1/2, mode='nearest')
        
        self.avg_pool = [nn.AvgPool2d(2**(i-1), stride = 2 ** (i-1)) for i in range(1,6+1)]
           
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, im, z, k, device):
        
        

        im1 = self.avg_pool[0](im)
        im2 = self.avg_pool[1](im)
        im3 = self.avg_pool[2](im)
        im4 = self.avg_pool[3](im)
        im5 = self.avg_pool[4](im)
        im6 = self.avg_pool[5](im)


        z6 = torch.cat((im6, z[5] * k), 1)
        z5 = torch.cat((im5, z[4] * k), 1)
        z4 = torch.cat((im4, z[3] * k), 1)
        z3 = torch.cat((im3, z[2] * k), 1)
        z2 = torch.cat((im2, z[1] * k), 1)
        z1 = torch.cat((im1, z[0] * k), 1)
        
        
        out_z6 = self.convblock6(z6)
        z6 = z6.to('cpu') if device == 'cuda' else z6
        out_z5 = self.convblock5(z5)
        z5 = z5.to('cpu') if device == 'cuda' else z5
        
        out = self.j56(out_z6, out_z5)
        out_z5 = out_z5.to('cpu') if device == 'cuda' else out_z5
        out_z6 = out_z6.to('cpu') if device == 'cuda' else out_z6
        
        
        out = self.joined_cb1(out)
        
        
        out_z4 = self.convblock4(z4)
        z4 = z4.to('cpu') if device == 'cuda' else z4
        
        out = self.j45(out, out_z4)
        
        out_z4 = out_z4.to('cpu') if device == 'cuda' else out_z4
        
        out = self.joined_cb2(out)
                
        out_z3 = self.convblock3(z3)
        z3 = z3.to('cpu') if device == 'cuda' else z3
        
        out = self.j34(out, out_z3)
        out_z3 = out_z3.to('cpu') if device == 'cuda' else out_z3
        
        out = self.joined_cb3(out)    

        out_z2 = self.convblock2(z2)
        z2 = z2.to('cpu') if device == 'cuda' else z2
        
        out = self.j23(out, out_z2)
        out_z2 = out_z2.to('cpu') if device == 'cuda' else out_z2
        
        out = self.joined_cb4(out)

        out_z1 = self.convblock1(z1)
        
        
        z1 = z1.to('cpu') if device == 'cuda' else z1
        
        
        out = self.j12(out, out_z1)
        out_z1 = out_z1.to('cpu') if device == 'cuda' else out_z1
        
        out = self.joined_cb5(out)
        
        return self.end_block(out) #self.end_relu(self.end_batch(self.end_conv(out)))
        
        