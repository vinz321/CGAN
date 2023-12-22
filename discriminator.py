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

def half_size(in_ch, out_ch):
    return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq=nn.Sequential(
            #256
            half_size(6,128), #128
            half_size(128,256), #64
            half_size(256,256), #32
            half_size(256,512), #16
            half_size(512,512), #8
            half_size(512,1024), #4
            half_size(1024,1024), #2
            nn.Conv2d(1024, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    

    def forward(self, in_data):
        x=self.seq(in_data)
        return x