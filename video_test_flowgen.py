import os
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import functional as transformF
import torchvision.utils as vutils
from torcheval.metrics import PeakSignalNoiseRatio,FrechetInceptionDistance
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




# parser=argparse.ArgumentParser()

# # parser.add_argument("model", choices=('unet', 'flowgen'))
# # parser.add_argument("--modelpath", type=str, default='.\\neverStreets.pth.tar')
# # parser.add_argument("--dsorder", type=str, default='d,dns')

# # args=parser.parse_args()
# dataset=DNSDatasetCustom(os.path.dirname(os.path.realpath(__file__)) + '\\dataset\\Cam 11', set_order=args.dsorder, depth_grayscale=args.model=='unet')

# if args.model=='unet':
#     model=unet
# if args.model=='flowgen':
#     model=flowgen

# print(PATH)
# print("Loading...")
# try:
#     state = torch.load(os.path.join(PATH,'.\\models\\neverStreetsUnet.pth.tar'))
#     if state:
#         unet.load_state_dict(state["Generator"])
#         print(dict(state).keys())
#         print(state['epoch'])
# except:
#     print("Error in load")

# try:
#     state = torch.load(os.path.join(PATH,args.modelpath))
#     if state:
#         model.load_state_dict(state["Generator"])
#         print(dict(state).keys())
#         print(state['epoch'])
# except:
#     print("Error in load")

while True:
    print("Input camera number:")
    cam_num=input()
    if(cam_num.isdigit() and "Cam "+cam_num in os.listdir( os.path.dirname(os.path.realpath(__file__)) + "\\dataset")):
        dataset=DNSDatasetCustom(os.path.dirname(os.path.realpath(__file__)) + '\\dataset\\Cam 11', set_order='d,dns', depth_grayscale=False)    

    print("Input model name 1")
    model_name=input()
    print("Input model name 2")
    model_name2=input()

    flowgen1=FlowGen(unet_pretrained='.\\models\\neverStreetsUnet.pth.tar').cuda().eval()
    flowgen1.reload_pretrained()
    flowgen2=FlowGen(unet_pretrained='.\\models\\neverStreetsUnet.pth.tar').cuda().eval()
    flowgen2.reload_pretrained()
    unet=Unet(7).cuda().eval()

    try:
        state = torch.load(os.path.join(PATH,'.\\models\\'+model_name))
        if state:
            flowgen1.load_state_dict(state["Generator"])
            print(dict(state).keys())
            print(state['epoch'])
            
    except:
        print("Error in load")

    try:
        state = torch.load(os.path.join(PATH,'.\\models\\neverStreetsUnet.pth.tar'))
        if state:
            unet.load_state_dict(state["Generator"])
            print(dict(state).keys())
            print(state['epoch'])
    except:
        print("Error in load")

    try:
        state = torch.load(os.path.join(PATH,'.\\models\\'+model_name2))
        if state:
            flowgen2.load_state_dict(state["Generator"])
            print(dict(state).keys())
            print(state['epoch'])
            break
            
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

psnr1_ls=[]
psnr2_ls=[]
psnr_unet_ls=[]

def cat_imgs(current_frame:torch.Tensor, inputs):
    # print("current_frame: ",current.shape)
    # cat_img=torch.cat([current_frame.unsqueeze(1), inputs.cuda()],1)

    d1=inputs[0:3,:,:]
    d2=inputs[3:6,:,:]
    n2=inputs[6:9,:,:]
    s2=inputs[9:12,:,:]
    cat_img=torch.cat((d1,current_frame,d2,s2,n2 ), 0).cuda()

    


    #cat_img=torch.stack((d1,current,d2,s2),0).cuda()
    
    # print("cat imgs ",cat_img.shape)
    return cat_img

with torch.no_grad():
    #current=unet(dataset[0][0][5:12].unsqueeze(0).cuda()).squeeze()
    current1=dataset[0][1].cuda().squeeze()
    current2=dataset[0][1].cuda().squeeze()
    psnr1=PeakSignalNoiseRatio()
    psnr2=PeakSignalNoiseRatio()
    psnr_unet=PeakSignalNoiseRatio()

    while(True):
        plt.figure(1)
        plt.clf()
        current=unet(dataset[i][0][5:12].unsqueeze(0).cuda()).squeeze()
        current1=flowgen1(cat_imgs(current1.squeeze(),dataset[i][0].cuda()).unsqueeze(0))[0]
        current2=flowgen2(cat_imgs(current2.squeeze(),dataset[i][0].cuda()).unsqueeze(0))[0]

        psnr1.update(current1.unsqueeze(0), dataset[i][1].cuda().unsqueeze(0))
        psnr2.update(current2.unsqueeze(0), dataset[i][1].cuda().unsqueeze(0))
        psnr_unet.update(current.unsqueeze(0), dataset[i][1].cuda().unsqueeze(0))

        v1=psnr1.compute().item()
        v2=psnr2.compute().item()
        vunet=psnr_unet.compute().item()

        psnr1_ls.append(v1)
        psnr2_ls.append(v2)
        psnr_unet_ls.append(vunet)
        print(current.shape)
        plt.subplot(231)
        plt.title("MPE")
        plt.imshow( to_image(current1.clip(0,1).cpu()))
        plt.subplot(232)
        plt.title("Kullback-Liebler")
        plt.imshow( to_image(current2.clip(0,1).cpu()))
        plt.subplot(233)
        plt.title("UNet")
        plt.imshow( to_image(current.clip(0,1).cpu()))
        #plt.imshow( to_image(dataset_train[i][1].unsqueeze(0).cuda()[0].cpu()) )
        plt.pause(0.0001)
        sleep(.0001)
        i+=1

        if i>=len(dataset):
            plt.clf()
            plt.ioff()
            plt.subplot(221)
            plt.title("PSNR model 1")
            plt.plot(psnr1_ls)
            plt.subplot(222)
            plt.title("PSNR model 2")
            plt.plot(psnr2_ls)
            plt.subplot(223)
            plt.title("PSNR model unet")
            plt.plot(psnr_unet_ls)
            plt.show()
            break

