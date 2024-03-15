import os
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import functional as transformF
import torchvision.utils as vutils
from torchvision.io import read_image,ImageReadMode
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from PIL import Image
from copy import deepcopy
from UNet import Unet
import numpy as np
import matplotlib.pyplot as plt 

PATH="C:\\Users\\vicin\Desktop\\PoliTo\\ML_for_CV\\CGAN"
BATCH_SIZE=16

unet=Unet(7).cuda()

if "neverStreets" in os.listdir(PATH):
  print("Loading...")
  try:
    state = torch.load(PATH+"/neverStreets")
    if state:
      unet.load_state_dict(state["Generator"])
  except:
    print("Error in load")

class DNSDataset(Dataset):
    def __init__(self, img_dir, seg=True, norm=True, depth=True, with_previous=False, transforms=None, target_transforms=None):
        self.img_dir=img_dir
        self.imgs_depth=[]
        self.imgs_norm=[]
        self.imgs_seg=[]
        self.imgs_col=[]
        self.norm=norm
        self.seg=seg
        self.depth=depth
        self.with_previous=with_previous

        if depth:
            self.imgs_depth.extend([self.img_dir + '\\Depth\\'+i for i in os.listdir(self.img_dir+'\\Depth') if i.endswith('.png')])
        if norm:
            self.imgs_norm.extend([self.img_dir + '\\Normal\\'+i for i in os.listdir(self.img_dir+'\\Normal') if i.endswith('.png')])
        if seg:
            self.imgs_seg.extend([self.img_dir + '\\Segmentation\\'+i for i in os.listdir(self.img_dir+'\\Segmentation') if i.endswith('.png')])
        
        self.imgs_col.extend([self.img_dir + '\\Default\\'+i for i in os.listdir(self.img_dir+'\\Default') if i.endswith('.png')])
        
        self.transforms=transforms
        self.target_transforms=target_transforms
    def __getitem__(self, index) -> torch.Tensor:
        tensors=[]
        if self.with_previous:
            prev_frame=transformF.to_tensor(Image.open(self.imgs_col[index]))[0:3]
            tensors.append(prev_frame)
            index=index+1

        if self.depth:
            depth=transformF.to_tensor(transformF.to_grayscale(Image.open(self.imgs_depth[index])))
            tensors.append(depth)
        if self.norm:
            norm=transformF.to_tensor(Image.open(self.imgs_norm[index]))[0:3]
            tensors.append(norm)
        if self.seg:
            seg=transformF.to_tensor(Image.open(self.imgs_seg[index]))[0:3]
            tensors.append(seg)
        col=transformF.to_tensor(Image.open(self.imgs_col[index]))[0:3]
        
        
        
        return torch.cat(tensors,0), col
        
    def __len__(self):
        if self.with_previous:
            return len(self.imgs_col)-1
        return len(self.imgs_col)
    
dataset_train=DNSDataset(PATH+'\\dataset\\Cam 4',norm=False)

plt.ion()
plt.show()
i=0
to_image=transforms.ToPILImage()

def cat_imgs(current_frame, inputs):
    return torch.cat([current_frame, inputs.cuda()],0)

with torch.no_grad():
    current=dataset_train[0][1].cuda()
    while(True):
        plt.figure(1)
        plt.clf()

        current=unet(cat_imgs(current,dataset_train[i][0]).unsqueeze(0))[0]
        # print(current.shape)
        plt.imshow( to_image(current.cpu()) )
        #plt.imshow( to_image(dataset_train[i][1].unsqueeze(0).cuda()[0].cpu()) )
        plt.pause(0.0001)
        sleep(.0001)
        i+=1

