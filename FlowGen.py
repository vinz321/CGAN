from argparse import ArgumentParser, Namespace
from torchsummary import summary
import torch.nn as nn
import torch
from torchvision.transforms import functional as transformF
from torch.nn.functional import grid_sample
import numpy as np
from PIL import Image
import os
from CGAN.flownet2.models import FlowNet2
from CGAN.flownet2.networks.resample2d_package.resample2d import Resample2d
from CGAN.flownet.models.FlowNetS import FlowNetS
from UNet import Unet
import matplotlib.pyplot as plt

# parser=ArgumentParser()

# parser.add_argument("--rgb_max", type=float, default = 255.)
# parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')

# args=parser.parse_args()




class FlowGen(nn.Module):
    def __init__(self, flow_mul:float =7., flow_mul_train=False, flownet_bn:bool=False ):
        super(FlowGen, self).__init__()
        self.args=Namespace(rgb_max=1., fp16=False)
        # self.resample=Resample2d()
        self.flow_mul=torch.scalar_tensor(flow_mul,requires_grad=flow_mul_train).cuda()


        self.flownet1=FlowNetS(flownet_bn)
        dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'\\flownet\\pretrained\\flownets_EPE1.951.pth.tar')
        print(dict.keys())
        self.flownet1.load_state_dict(dict["state_dict"])       

        # dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'\\flownet2\\pretrainedmodels\\FlowNet2.pth.tar')

        # self.flownet2=FlowNet2(self.args, div_flow=200)
        # self.flownet2.load_state_dict(dict["state_dict"])
        # self.flownet2.requires_grad_(False)
        #self.flownet2.flownetfusion.deconv1.requires_grad_(True)

        self.unet=Unet(7)

    def reload_flownet(self):
        dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'\\flownet\\pretrained\\flownets_EPE1.951.pth.tar')
        print(dict.keys())
        self.flownet1.load_state_dict(dict["state_dict"])       


    def forward(self, x):
        
        depth_1=x[:,0:3,:,:]
        img_1=x[:,3:6,:,:]
        depth_2=x[:,6:9,:,:]
        seg_2=x[:,9:12,:,:]

        # x1=x1.expand((-1,3,-1,-1,-1))
        # x2=x2.expand((-1,3,-1,-1,-1))

        cat=torch.cat([depth_1,depth_2], 1)
        
        flow=self.flownet1(cat)
        # zero_layer=torch.zeros([1,1,64,64]).cuda()

        w=torch.arange(0,64,1)
        h=torch.arange(0,64,1)

        w=w.view([1,1,-1,1]).repeat([1,64,1,1]).cuda() 
        h=h.view([1,-1,1,1]).repeat([1,1,64,1]).cuda()

        add_dim=(-flow[0].permute(0,2,3,1)*self.flow_mul+ 2*torch.cat([w,h], 3))/63 -1

        upsampled=torch.nn.functional.interpolate(add_dim.permute(0,3,1,2), [256,256], antialias='true', mode='bilinear')
        warped_img=grid_sample(img_1, upsampled.permute(0,2,3,1))

        depth_2=x[:,6,:,:].unsqueeze(1)
        cat_imgs=torch.cat([warped_img,depth_2,seg_2],1)

        o=self.unet(cat_imgs)
        #warped_flow=self.resample(x1.squeeze(2), flow)
        # torch.cat([flow[0],zero_layer],1)
        return o


depth_1=transformF.to_tensor(Image.open(os.path.dirname(os.path.realpath(__file__))+'\\dataset\\Cam 1\\Depth\\Image0020.png'))[0:3,:,:].unsqueeze(1)
img_1=transformF.to_tensor(Image.open(os.path.dirname(os.path.realpath(__file__))+'\\dataset\\Cam 1\\Default\\0020.png'))[0:3,:,:].unsqueeze(1)

depth_2=transformF.to_tensor(Image.open(os.path.dirname(os.path.realpath(__file__))+'\\dataset\\Cam 1\\Depth\\Image0050.png'))[0:3,:,:].unsqueeze(1)
seg_2=transformF.to_tensor(Image.open(os.path.dirname(os.path.realpath(__file__))+'\\dataset\\Cam 1\\Segmentation\\0050.png'))[0:3,:,:].unsqueeze(1)

print("Im1: "+str(depth_1.shape))
combined_img=torch.cat([depth_1,img_1,depth_2,seg_2], 1 ).unsqueeze(0).cuda()

print("Combined: "+str(combined_img.shape))
flow=FlowGen().cuda()


### Test FlowNet e Warp ###
# warped=flow(combined_img)
# upsampled=torch.nn.functional.interpolate(warped[1].permute(0,3,1,2), [256,256], antialias='true', mode='bilinear')
# print(upsampled.shape)
# warped_img=grid_sample(img_1.cuda(), upsampled.permute(0,2,3,1))
# print(warped)

# plt.subplot(221)
# plt.imshow(depth_1.squeeze(1).permute(1,2,0).cpu())
# plt.subplot(222)
# plt.imshow(depth_2.squeeze(1).permute(1,2,0).cpu())
# plt.subplot(223)
# plt.imshow(warped[0].squeeze(0).permute(1,2,0).detach().cpu().numpy())
# plt.subplot(224)
# plt.imshow(warped_img.squeeze(0).permute(1,2,0).detach().cpu().numpy())
# plt.show()

### FINE TEST ###

summary(flow, (12,256,256))
