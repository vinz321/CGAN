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
from FlowGenV2 import FlowGen
from datasets import DNSDatasetCustom, full_dns_dataset
import numpy as np
import matplotlib.pyplot as plt 
import argparse 
import keyboard
import random
unet_dns=Unet(7)
unet_ds=Unet(4)

PATH=os.path.dirname(os.path.realpath(__file__))

try:
    state = torch.load(os.path.join(PATH,'.\\models\\neverStreetsUnet.pth.tar'))
    if state:
        unet_dns.load_state_dict(state["Generator"])
        print(dict(state).keys())
        print(state['epoch'])
except:
    print("Error in load")

try:
    state = torch.load(os.path.join(PATH,'.\\models\\neverStreetsUnetNoNorm.pth.tar'))
    if state:
        unet_ds.load_state_dict(state["Generator"])
        print(dict(state).keys())
        print(state['epoch'])
except:
    print("Error in load")

dataset_train=full_dns_dataset(PATH+'\\dataset', 'Cam ',1,8, ',ds',transforms=None, co_transform=None)
dataset_train_dns=full_dns_dataset(PATH+'\\dataset', 'Cam ',1,8, ',dns',transforms=None, co_transform=None)
dataset_test=full_dns_dataset(PATH+'\\dataset', 'Cam ',10,12, ',ds',transforms=None, co_transform=None)
dataset_test_dns=full_dns_dataset(PATH+'\\dataset', 'Cam ',10,12, ',dns',transforms=None, co_transform=None)

while True:
    print("Inserisci \"ts\" per generare 2 immagini dal test set, \"tr\"  per generare 2 immagini dal train set\nPremi Esc per uscire")
    keyboard.add_hotkey('esc', lambda: quit())
    s=input()

    if s.lower()=='ts':
        rand_idx=random.randint(0,len(dataset_test)-1)
        rand_data_ds=dataset_test[rand_idx][0][2:]
        rand_data_dns=dataset_test_dns[rand_idx][0][2:]
        gen_img_dns=unet_dns(rand_data_dns.unsqueeze(0))[0]
        gen_img_ds=unet_ds(rand_data_ds.unsqueeze(0))[0]
        plt.subplot(121)
        plt.title("No normal")
        plt.imshow(gen_img_ds.detach().permute([1,2,0]))
        plt.subplot(122)
        plt.title("With normal")
        plt.imshow(gen_img_dns.detach().permute([1,2,0]))
        plt.show()
    elif s.lower()=='tr':
        rand_idx=random.randint(0,len(dataset_test)-1)
        rand_data_ds=dataset_train[rand_idx][0][2:]
        rand_data_dns=dataset_train_dns[rand_idx][0][2:]
        gen_img_dns=unet_dns(rand_data_dns.unsqueeze(0))[0]
        gen_img_ds=unet_ds(rand_data_ds.unsqueeze(0))[0]
        plt.subplot(121)
        plt.title("No normal")
        plt.imshow(gen_img_ds.detach().permute([1,2,0]))
        plt.subplot(122)
        plt.title("With normal")
        plt.imshow(gen_img_dns.detach().permute([1,2,0]))
        plt.show()



