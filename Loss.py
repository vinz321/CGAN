import torch
import torch.nn as nn

# https://arxiv.org/pdf/2010.13085.pdf  coherent loss
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8902992

class LossCustom(nn.Module): 
    def __init__(self):
        super(LossCustom, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.kl_loss = nn.KLDivLoss()
    
    def wasserstein_loss(self, output, target):
        return torch.mean(output*target)
        
    def forward(self, output, target):
        l1 = self.l1_loss(output, target)
        kl = self.kl_loss(output, target)
        wganl = self.wasserstein_loss(output, target)
        return wganl + l1 + kl 
   
    
class LossCustomDisc(nn.Module):
    def __init__(self):
        super(LossCustomDisc, self).__init__()
    
    def wasserstein_loss(self, output, target):
        return torch.mean(output*target)
    
    def forward(self, output, target):
        wganl = self.wasserstein_loss(output, target)
        return wganl
        

if __name__=='__main__':
    lc=LossCustom()
    t1=torch.scalar_tensor(1)
    t2=torch.scalar_tensor(2)
    
    lc(t1,t2)
    