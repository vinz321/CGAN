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
from datasets import DNSDatasetCustom
import numpy as np
import matplotlib.pyplot as plt 
import argparse 
PATH=os.path.dirname(os.path.realpath(__file__))
BATCH_SIZE=16


flowgen=FlowGen(unet_pretrained='.\\models\\neverStreetsUnet.pth.tar').cuda().eval()
flowgen.reload_pretrained()

unet=Unet(7).cuda().eval()

parser=argparse.ArgumentParser()

parser.add_argument("model", choices=('unet', 'flowgen'))
parser.add_argument("--modelpath", type=str, default='.\\neverStreets.pth.tar')
parser.add_argument("--dsorder", type=str, default='d,dns')

args=parser.parse_args()
dataset=DNSDatasetCustom(os.path.dirname(os.path.realpath(__file__)) + '\\dataset\\Cam 11', set_order=args.dsorder, depth_grayscale=args.model=='unet')

if args.model=='unet':
    model=unet
if args.model=='flowgen':
    model=flowgen

print(PATH)
print("Loading...")
try:
    state = torch.load(os.path.join(PATH,'.\\models\\neverStreetsUnet.pth.tar'))
    if state:
        unet.load_state_dict(state["Generator"])
        print(dict(state).keys())
        print(state['epoch'])
except:
    print("Error in load")

try:
    state = torch.load(os.path.join(PATH,args.modelpath))
    if state:
        model.load_state_dict(state["Generator"])
        print(dict(state).keys())
        print(state['epoch'])
except:
    print("Error in load")

print(dataset[0][0].shape)
plt.subplot(2,2,1)
plt.imshow(dataset[0][0][0:3].cuda().squeeze().permute([1,2,0]).cpu().detach().numpy())
plt.subplot(2,2,2)
plt.imshow(dataset[0][0][3:6].squeeze().permute([1,2,0]).cpu().numpy())
plt.subplot(2,2,3)
plt.imshow(dataset[0][0][6:9].squeeze().permute([1,2,0]).cpu().numpy())
plt.subplot(2,2,4)
plt.imshow(dataset[0][0][9:12].squeeze().permute([1,2,0]).cpu().numpy())
plt.show()
plt.ion()
plt.show()
i=0
to_image=transforms.ToPILImage()

def cat_imgs(current_frame:torch.Tensor, inputs):
    # print("current_frame: ",current.shape)
    # cat_img=torch.cat([current_frame.unsqueeze(1), inputs.cuda()],1)

    if args.model=='flowgen':
        d1=inputs[0:3,:,:]
        d2=inputs[3:6,:,:]
        n2=inputs[6:9,:,:]
        s2=inputs[9:12,:,:]
        cat_img=torch.cat((d1,current,d2,s2,n2 ), 0).cuda()
    elif args.model=='unet':
        cat_img=inputs
    

    #cat_img=torch.stack((d1,current,d2,s2),0).cuda()
    
    # print("cat imgs ",cat_img.shape)
    return cat_img

with torch.no_grad():
    if args.model=='flowgen':
        current=unet(dataset[0][0][5:12].unsqueeze(0).cuda()).squeeze()
    else:
        current=dataset[0][1].cuda()
    while(True):
        plt.figure(1)
        plt.clf()

        current=model(cat_imgs(current.squeeze(),dataset[i][0].cuda()).unsqueeze(0))[0]
        # print(current.shape)
        plt.imshow( to_image(current.clip(0,1).cpu()))
        
        #plt.imshow( to_image(dataset_train[i][1].unsqueeze(0).cuda()[0].cpu()) )
        plt.pause(0.0001)
        sleep(.0001)
        i+=1

