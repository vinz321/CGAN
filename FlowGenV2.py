from argparse import ArgumentParser, Namespace
from torchsummary import summary
import torch.nn as nn
import torch
from torchvision.transforms import functional as transformF
from torch.nn.functional import grid_sample
import numpy as np
from PIL import Image
import os
from CGAN.flownet.models.FlowNetS import FlowNetS
from UNet import Unet
import matplotlib.pyplot as plt

# parser=ArgumentParser()

# parser.add_argument("--rgb_max", type=float, default = 255.)
# parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')

# args=parser.parse_args()

PATH=os.path.dirname(os.path.realpath(__file__))



class FlowGen(nn.Module):
    def __init__(self, flow_mul:float =7., flow_mul_train=False, flownet_bn:bool=False, 
                 unet_pretrained='.\\models\\neverStreets', flownet_pretrained='.\\flownet\\pretrained\\flownets_EPE1.951.pth.tar', std_dev=.01, mean=0.02 ):
        global PATH
        super(FlowGen, self).__init__()
        self.std_dev=std_dev
        self.mean=mean
        self.args=Namespace(rgb_max=1., fp16=False)
        self.flownet_pretrained=flownet_pretrained
        self.unet_pretrained=unet_pretrained
        # self.resample=Resample2d()
        self.flow_mul=torch.scalar_tensor(flow_mul,requires_grad=flow_mul_train).cuda()


        self.flownet1=FlowNetS(flownet_bn)  
        self.flownet1.requires_grad_(False)
        # dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'\\flownet2\\pretrainedmodels\\FlowNet2.pth.tar')

        # self.flownet2=FlowNet2(self.args, div_flow=200)
        # self.flownet2.load_state_dict(dict["state_dict"])
        #self.flownet2.flownetfusion.deconv1.requires_grad_(True)
        self.unet=Unet(7)
        self.unet.requires_grad_(False)

        self.unet2=Unet(3)
        
    

    def reload_pretrained(self):
        dict = torch.load(os.path.join(PATH, self.flownet_pretrained))
        print(dict.keys())
        self.flownet1.load_state_dict(dict["state_dict"])  

        dict = torch.load(os.path.join(PATH, self.unet_pretrained))
        print(dict.keys())
        self.unet.load_state_dict(dict["Generator"])       


    def forward(self, x):
        
        depth_1=x[:,0:3,:,:]
        img_1=x[:,3:6,:,:]
        
        depth_2=x[:,6:9,:,:]
        seg_2=x[:,9:12,:,:]
        norm_2=x[:,12:15,:,:]

        # x1=x1.expand((-1,3,-1,-1,-1))
        # x2=x2.expand((-1,3,-1,-1,-1))

        cat=torch.cat([depth_1,depth_2], 1)
        
        flow=self.flownet1(cat)
        # zero_layer=torch.zeros([1,1,64,64]).cuda()

        w=torch.arange(0,64,1)
        h=torch.arange(0,64,1)

        w=w.view([1,1,-1,1]).repeat([1,64,1,1]).cuda() 
        h=h.view([1,-1,1,1]).repeat([1,1,64,1]).cuda()

        if(self.training):
            add_dim=(-flow[0].permute(0,2,3,1)*self.flow_mul+ 2*torch.cat([w,h], 3))/63 -1
        else:
            add_dim=(-flow.permute(0,2,3,1)*self.flow_mul + 2*torch.cat([w,h], 3))/63 -1
        upsampled=torch.nn.functional.interpolate(add_dim.permute(0,3,1,2), [256,256], antialias='true', mode='bilinear')
        warped_img=grid_sample(img_1, upsampled.permute(0,2,3,1))

        warped_img+=(torch.randn(warped_img.size())*self.std_dev+self.mean).cuda()

        depth_2=x[:,6,:,:].unsqueeze(1)
        cat_imgs=torch.cat([depth_2, norm_2, seg_2],1)

        o=self.unet(cat_imgs)
        
        # w1*Fi +w2* Warp(Fi-1) 
        o2=self.unet2((o*3+warped_img)/4)

        return o2


if __name__=='__main__':

    depth_1=transformF.to_tensor(Image.open(os.path.dirname(os.path.realpath(__file__))+'\\dataset\\Cam 1\\Depth\\Image0020.png'))[0:3,:,:].unsqueeze(1)
    img_1=transformF.to_tensor(Image.open(os.path.dirname(os.path.realpath(__file__))+'\\dataset\\Cam 1\\Default\\0020.png'))[0:3,:,:].unsqueeze(1)

    depth_2=transformF.to_tensor(Image.open(os.path.dirname(os.path.realpath(__file__))+'\\dataset\\Cam 1\\Depth\\Image0050.png'))[0:3,:,:].unsqueeze(1)
    seg_2=transformF.to_tensor(Image.open(os.path.dirname(os.path.realpath(__file__))+'\\dataset\\Cam 1\\Segmentation\\0050.png'))[0:3,:,:].unsqueeze(1)

    print("Im1: "+str(depth_1.shape))
    combined_img=torch.cat([depth_1,img_1,depth_2,seg_2], 1 ).unsqueeze(0).cuda()

    print("Combined: "+str(combined_img.shape))
    flow=FlowGen().cuda()
    summary(flow, (15,256,256))