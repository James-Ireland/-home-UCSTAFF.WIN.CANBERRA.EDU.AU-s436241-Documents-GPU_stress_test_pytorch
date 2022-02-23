#!/usr/bin/env python3 
import os
import argparse 
import torchvision 
import torchvision.transforms as T 
import torch
import torch.utils.data
from PIL import Image
import numpy as np 
from tqdm import tqdm 

class Synthetic_Fake_Image_Stress_Test(torch.utils.data.Dataset):
    def __init__(self,  image_width=1080, image_height=1920, image_channels=3, numOfFakeImages=10000, transforms=T.ToTensor()): 
        self.dataset_length = numOfFakeImages
        self.w = image_width
        self.h = image_height
        self.c = image_channels
        self.transforms = transforms
        
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        numpy_image = np.zeros([ self.w, self.h, self.c])
        image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, 1


def PytorchStressTest(opt): 
    model = torchvision.models.resnet50(pretrained=False) 
    print('Loading data ...............')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Synthetic_Fake_Image_Stress_Test( opt.image_width, opt.image_height, opt.image_channels, opt.numOfFakeImages)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    print('Starting stress test ...............')
    model.to(device)   
    for i in tqdm(range(0, opt.numOfIterations), position=0, leave=True):  
        for images, GTs in tqdm(data_loader, position=0, leave=True):  
            preds = model(images.to(device))   
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PytorchStressTest.py')
    parser.add_argument('--batch_size', type=int, default=3, help='size of each image batch')
    parser.add_argument('--numOfIterations', type=int, default=1, help='size of each image batch')
    parser.add_argument('--image_width', type=int, default=1080, help='width of each image ')
    parser.add_argument('--image_height', type=int, default=1920, help='height of each image ')
    parser.add_argument('--image_channels', type=int, default=3, help='Number of channels in image (RGB == 3)')
    parser.add_argument('--numOfFakeImages', type=int, default=10000, help='Number of fake images per iteration') 
    opt = parser.parse_args()
    PytorchStressTest(opt) 