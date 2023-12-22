
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from pathlib import Path
from configparser import ConfigParser

PATH=Path(__file__).parent.resolve()
parser=ConfigParser()
parser.read(str(PATH)+'\\config.ini')
VEC_SIZE=int(parser.get('Default', 'VEC_SIZE'))
BATCH_SIZE=int(parser.get('Default', 'BATCH_SIZE'))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential( #256
            self.newSeq(3, 128), #128
            self.newSeq(128, 128), #64
            self.newSeq(128, 256), #32
            self.newSeq(256, 256), #16
            self.newSeq(256, 512), #8
            self.newSeq(512, 512), #4
            #self.newSeq(512, 512), #2
            nn.Conv2d(512, VEC_SIZE, 4, 2, 1),#2
            nn.ReLU()
        )

    def newSeq(self, in_ch, out_ch):
        seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01)
        )
        return seq
 
    def forward(self, x):
        output = self.seq(x)
        return output



