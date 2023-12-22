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

print(VEC_SIZE)

def double_size(in_ch, out_ch):
    return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq=nn.Sequential(
            nn.ConvTranspose2d(VEC_SIZE, 1024, 4, 2, 1, bias=False), #2
            nn.LeakyReLU(0.2, True),
            #4x4
            double_size(1024,1024), #8
            double_size(1024,512), #16
            double_size(512,512), #32
            double_size(512,256), #64
            double_size(256,256), #128
            double_size(256,128), #256
            
            nn.Conv2d(128,3,1,1, 0, bias=False),
            nn.Tanh()
            

        )

    

    def forward(self, in_data):
        x=self.seq(in_data)
        return x