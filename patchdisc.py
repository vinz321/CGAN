import torch
from torch import nn,functional, utils

### V1 ###
class PatchDiscriminator(nn.Module):
    def __init__(self, in_chan=3, patch_size=70):
        super().__init__()

        size=patch_size
        channels=in_chan
        next_channels=64

        # self.seq=nn.Sequential(self.half_size(in_chan,64), #256-128
        #                        self.half_size(64,128),  #64
        #                        self.half_size(128,256), #32
        #                        self.half_size(256,512), #16
        #                        nn.Conv2d(512,1,1,1,bias=False), #16
        #                        nn.Sigmoid())
        layers=[]
        while(size>1):
            if (size%2)==0:
                layers.append(self.half_size(channels, next_channels))
                size/=2
                channels=next_channels
                next_channels*=2
            else:
                layers.append(self.minus1_size(channels, next_channels))
                size-=1
                channels=next_channels
                
        layers.extend([nn.Conv2d(channels,1,1,1,bias=False),
                               nn.Sigmoid()])

        self.seq=nn.Sequential(*layers)
                
        

                               

    def half_size(self, in_chan, out_chan, norm=True):
        if norm:
            return nn.Sequential(nn.Conv2d(in_chan, out_chan, 4,2,1, bias=False),
                            nn.BatchNorm2d(out_chan),
                            nn.LeakyReLU(0.2, True))
        else:
            return nn.Sequential(nn.Conv2d(in_chan, out_chan, 4,2,1, bias=False),
                            nn.LeakyReLU(0.2, True))
        

    def minus1_size(self, in_chan, out_chan, norm=True):
        if norm:
            return nn.Sequential(nn.Conv2d(in_chan, out_chan, 4,1,1, bias=False),
                            nn.BatchNorm2d(out_chan),
                            nn.LeakyReLU(0.2, True))
        else:
            return nn.Sequential(nn.Conv2d(in_chan, out_chan, 4,1,1, bias=False),
                            nn.LeakyReLU(0.2, True))

    def forward(self, in_data):
        return self.seq(in_data)