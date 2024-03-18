from multiprocessing import freeze_support
import torch
from torch.utils.data import DataLoader
import os
import datasets

freeze_support()
cust_dataset=datasets.full_dns_dataset(os.path.abspath(os.path.dirname(__file__))+'\\dataset','Cam ', 1,3,'d,d', None,None, False, 'o', None, True)
loader=DataLoader(
        cust_dataset, batch_size=16,
        num_workers=4, pin_memory=True, shuffle=True)

if __name__=='__main__':
    freeze_support()
    for i,(input,target) in enumerate(loader):
        print(input.shape)