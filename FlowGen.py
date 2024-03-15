from argparse import ArgumentParser, Namespace
from torchsummary import summary
import torch.nn as nn
import torch
from CGAN.flownet2.models import FlowNet2
from CGAN.flownet2.networks.resample2d_package.resample2d import Resample2d

parser=ArgumentParser()

parser.add_argument("--rgb_max", type=float, default = 255.)
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')

args=parser.parse_args()


class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()
        self.args=Namespace(rgb_max=255., fp16=False)
        self.preconv=nn.Conv2d(1,3, 4,2,1)

        self.flownet2=FlowNet2(self.args)
        


    def forward(self, x):
        print(x.shape)
        x1=self.preconv(x[:,:,0,:,:]).unsqueeze(2)
        x2=self.preconv(x[:,:,1,:,:]).unsqueeze(2)

        self.flownet2.requires_grad_(False)
        self.flownet2.flownetfusion.deconv1.requires_grad_(True)

        print(x1.shape)
        print(x2.shape)
        cat=torch.cat([x1,x2], 2)
        print(cat.shape)
        return self.flownet2(cat)


class FlowNoConv(nn.Module):
    def __init__(self):
        super(FlowNoConv, self).__init__()
        self.args=Namespace(rgb_max=255., fp16=False)
        self.resample=Resample2d()
        self.flownet2=FlowNet2(self.args)
        self.flownet2.requires_grad_(False)
        self.flownet2.flownetfusion.deconv1.requires_grad_(True)


    def forward(self, x):
        print(x.shape)
        x1=x[:,:,0,:,:].unsqueeze(2)
        x1=x1.expand((-1,3,-1,-1,-1))
        x2=x[:,:,1,:,:].unsqueeze(2)
        x2=x2.expand((-1,3,-1,-1,-1))

        

        cat=torch.cat([x1,x2], 2)
        flow=self.flownet2(cat)

        warped_flow=self.resample(x1.squeeze(2), flow)

        print(x1.shape)
        print(x2.shape)
        print(warped_flow.shape)
        return 


flow=FlowNoConv().cuda()
summary(flow, (1,2,384,512))
