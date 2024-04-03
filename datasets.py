import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.transforms import functional as transformF
from PIL import Image
from imageio import imread
class DNSDataset(Dataset):
    def __init__(self, img_dir, seg=True, norm=True, depth=True, with_previous=False, transforms=None, target_transforms=None):
        self.img_dir=img_dir
        self.imgs_depth=[]
        self.imgs_norm=[]
        self.imgs_seg=[]
        self.imgs_col=[]
        self.norm=norm
        self.seg=seg
        self.depth=depth
        self.with_previous=with_previous

        if depth:
            self.imgs_depth.extend([self.img_dir + '\\Depth\\'+i for i in os.listdir(self.img_dir+'\\Depth') if i.endswith('.png')])
        if norm:
            self.imgs_norm.extend([self.img_dir + '\\Normal\\'+i for i in os.listdir(self.img_dir+'\\Normal') if i.endswith('.png')])
        if seg:
            self.imgs_seg.extend([self.img_dir + '\\Segmentation\\'+i for i in os.listdir(self.img_dir+'\\Segmentation') if i.endswith('.png')])
        

        self.imgs_col.extend([self.img_dir + '\\Default\\'+i for i in os.listdir(self.img_dir+'\\Default') if i.endswith('.png')])
        
        self.transforms=transforms
        self.target_transforms=target_transforms
    def __getitem__(self, index) -> torch.Tensor:
        
        tensors=[]

        if self.with_previous:
            prev_frame=transformF.to_tensor(Image.open(self.imgs_col[0]))[0:3] #index invece che 0
            tensors.append(prev_frame)
            index=index+1

        if self.depth:
            depth=transformF.to_tensor(transformF.to_grayscale(Image.open(self.imgs_depth[index])))
            tensors.append(depth)
        if self.norm:
            norm=transformF.to_tensor(Image.open(self.imgs_norm[index]))[0:3]
            tensors.append(norm)
        if self.seg:
            seg=transformF.to_tensor(Image.open(self.imgs_seg[index]))[0:3]
            tensors.append(seg)
        col=transformF.to_tensor(Image.open(self.imgs_col[index]))[0:3]
        
        
        
        return torch.cat(tensors,0), col
        
    def __len__(self):
        if self.with_previous:
            return len(self.imgs_col)-1
        return len(self.imgs_col)
    

class DNSDatasetCustom(Dataset):
    def __init__(self, img_dir, seg=True, norm=True, depth=True, depth_grayscale=False, with_previous=True, set_order='dn,ds', 
                 transforms=None, target_transforms=None, co_transforms=None, 
                 stacked=True, output_type='c', std_io=False):
        self.img_dir=img_dir
        self.imgs_depth=[]
        self.imgs_norm=[]
        self.imgs_seg=[]
        self.imgs_col=[]
        self.imgs_of=[]
        self.set_order=set_order
        self.norm=norm
        self.seg=seg
        self.depth=depth
        self.with_previous=with_previous
        self.stacked=stacked
        self.output_type=output_type
        self.feature_loader=self.load_features
        self.std_io=std_io
        self.depth_grayscale=depth_grayscale
        if(std_io):
            self.feature_loader=self.load_features_stdio
        if depth:
            self.imgs_depth.extend([self.img_dir + '\\Depth\\'+i for i in os.listdir(self.img_dir+'\\Depth') if i.endswith('.png')])
        if norm:
            self.imgs_norm.extend([self.img_dir + '\\Normal\\'+i for i in os.listdir(self.img_dir+'\\Normal') if i.endswith('.png')])
        if seg:
            self.imgs_seg.extend([self.img_dir + '\\Segmentation\\'+i for i in os.listdir(self.img_dir+'\\Segmentation') if i.endswith('.png')])
        
        self.imgs_of.extend([self.img_dir + '\\Optical Flow\\'+i for i in os.listdir(self.img_dir+'\\Optical Flow') if i.endswith('.png')])
        
        self.imgs_col.extend([self.img_dir + '\\Default\\'+i for i in os.listdir(self.img_dir+'\\Default') if i.endswith('.png')])
        
        self.transforms=transforms
        self.target_transforms=target_transforms
        self.co_transforms=co_transforms
        
    def __getitem__(self, index) -> torch.Tensor:

        if(torch.is_tensor(index) or index is list):
            print("Index is a list")

        tensors=[]

        i1,i2=self.set_order.split(',',1)

        for l in i1:
            tensors.append(self.feature_loader(l, index))

        if self.with_previous:
            index+=1
            for l in i2:
                tensors.append(self.feature_loader(l, index))

        input=tensors
        target=self.feature_loader(self.output_type, index)
        if not self.std_io:
            input,target=(torch.cat(input,0), target)

            if self.co_transforms:
                temp=self.co_transforms(torch.cat((input, target),0))
                input,target=(temp[0:-3], temp[-3:])
                
        else:
            if self.co_transforms:
                input,target=self.co_transforms(input,target)

            if self.transforms:
                for i in range(0,len(input)):
                    input[i]=self.transforms(input[i])
            if self.target_transforms:
                target=self.target_transforms(target)
        # print(input.shape)
        return input,target

        
    def __len__(self):
        if self.with_previous:
            return len(self.imgs_col)-1
        return len(self.imgs_col)
    
    def load_features(self, type:str, index:int):
        if(type=='d'):
            if self.depth_grayscale:
                return transformF.to_tensor(transformF.to_grayscale(Image.open(self.imgs_depth[index])))
            return transformF.to_tensor(Image.open(self.imgs_depth[index]))[0:3]
        elif(type=='s'):
            return transformF.to_tensor(Image.open(self.imgs_seg[index]))[0:3]
        elif(type=='n'):
            return transformF.to_tensor(Image.open(self.imgs_norm[index]))[0:3]
        elif(type=='c'):
            return transformF.to_tensor(Image.open(self.imgs_col[index]))[0:3]
        elif(type=='o'):
            return transformF.to_tensor(Image.open(self.imgs_of[index]))[0:3]
    
    def load_features_stdio(self, type:str, index:int):
        if(type=='d'):
            return imread(self.imgs_depth[index]).astype(np.float32)[:,:,0:3]
        elif(type=='s'):
            return imread(self.imgs_seg[index]).astype(np.float32)[:,:,0:3]
        elif(type=='n'):
            return imread(self.imgs_norm[index]).astype(np.float32)[:,:,0:3]
        elif(type=='c'):
            return imread(self.imgs_col[index]).astype(np.float32)[:,:,0:3]
        elif(type=='o'):
            return imread(self.imgs_of[index]).astype(np.float32)[:,:,0:2]
        

class DNSDatasetFull(Dataset):
    def __init__(self, dir:str , subfolder_name:str=None ,from_folder:int=1, to_folder:int=7):
        dirs=os.listdir(dir)
        if(subfolder_name != None):
            dirs=[i for i in dirs if subfolder_name in i]
        
        dirs=[i for i in dirs if i[-1].isdigit() and int(i[-1]) in range(from_folder,to_folder+1)]


def full_dns_dataset( dir:str , subfolder_name:str=None ,from_folder:int=1, to_folder:int=7, set_order='dc,ds', transforms=None, target_transforms=None, stacked=True, output_type='c', co_transform=None, pil_image=False):
    dirs=os.listdir(dir)
    if(subfolder_name != None):
        dirs=[i for i in dirs if subfolder_name in i]
    
    dirs=[i for i in dirs if i[-1].isdigit() and int(i[-1]) in range(from_folder,to_folder+1)]

    datasets=[]
    for i in dirs:
        datasets.append(DNSDatasetCustom(dir+'\\'+i, set_order=set_order, transforms=transforms, target_transforms=target_transforms, co_transforms=co_transform, stacked=stacked,output_type=output_type, std_io=pil_image))

    
    return ConcatDataset(datasets)


if __name__=="__main__":
    dataset=DNSDatasetCustom(os.path.dirname(os.path.realpath(__file__)) + '\\dataset\\Cam 6', set_order='dc,dns', depth_grayscale=False)

    print(dataset[0])
