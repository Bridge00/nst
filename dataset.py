import torch.utils.data
#import pandas as pd
from csv import reader
import os
from PIL import Image
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.mat', '.npy',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Custom_Dataset(torch.utils.data.Dataset):

    def __init__(self, directory, transform = None):
        #self.data = pd.read_csv(file_path)
        
        #with open(file_path, 'r') as read_obj:
        #    csv_reader = reader(read_obj)
        #    self.data = [[float(val) for val in row] for row in list(csv_reader)]
        self.dir = directory
        self.data = [i for i in os.listdir(self.dir) if is_image_file(i)]

        self.transform = transform
        
        #print(self.data)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        
        path = os.path.join(self.dir, self.data[index])
        #print(path)
        img = Image.open(path) #.convert('RGB') ##.resize((self.loadSize,self.loadSize), Image.BICUBIC)
        #print(img.size)
        if self.transform is not None:
            img = self.transform(img)
            #img = torch.unsqueeze(img, 0)
        #print(img.size())
        return img