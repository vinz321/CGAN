import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from pathlib import Path
from configparser import ConfigParser

PATH=str(Path(__file__).parent.resolve())
parser=ConfigParser()
parser.read(PATH+'\\config.ini')

VEC_SIZE=int(parser.get('Default', 'VEC_SIZE'))
BATCH_SIZE=int(parser.get('Default', 'BATCH_SIZE'))


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01)
        )
    
    def forward(self, x):
        output = self.seq(x)
        return output
    
    

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.seq = nn.Sequential(
            nn.LazyConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, x):
        output = self.seq(x)
        return output
    



class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        #256
        self.enc1 = Encoder(3, 128) #128
        self.enc2 = Encoder(128, 128) #64
        self.enc3 = Encoder(128, 256) #32
        self.enc4 = Encoder(256, 256) #16
        self.enc5 = Encoder(256, 512) #8
        self.enc6 = Encoder(512, 512) #4
        self.conv = nn.Conv2d(512, VEC_SIZE, 4, 2, 1) #2
        self.relu = nn.ReLU()

        #2
        self.conv_t = nn.ConvTranspose2d(VEC_SIZE, 1024, 4, 2, 1, bias=False) #4
        self.l_relu = nn.LeakyReLU(0.2, True)
        self.dec1 = Decoder(1024, 1024) #8
        self.dec2 = Decoder(1024, 512) #16
        self.dec3 = Decoder(512, 512) #32
        self.dec4 = Decoder(512, 256) #64
        self.dec5 = Decoder(256, 256) #128
        self.dec6 = Decoder(256, 128) #256
        self.conv7 = nn.Conv2d(128, 3, 1, 1, 0, bias=False)
        self.tanh = nn.Tanh()

    def connection(x, y):
        return torch.cat((x, y))


    def forward(self,x):

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc5(x5)
        
        x = self.conv(x6)
        x = self.relu(x)
        x = self.conv_t(x)
        x = self.tanh(x)
        
        x = self.dec1(x)
        x = self.connection(x6, x)
        x = self.dec2(x)
        x = self.connection(x5, x)
        x = self.dec3(x)
        x = self.connection(x4, x)
        x = self.dec4(x)
        x = self.connection(x3, x)
        x = self.dec5(x)
        x = self.connection(x2, x)
        x = self.dec6(x)
        x = self.connection(x1, x)

        x = self.conv7(x)
        x = self.tanh(x)

        return x
