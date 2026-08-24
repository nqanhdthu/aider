import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t=int(abs((math.log2(channels)+b)/gamma))
        k=t if t%2 else t+1
        k=max(1,k)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=k,padding=(k-1)//2,bias=False)
    def forward(self,x):
        y=self.pool(x).squeeze(-1).transpose(-1,-2)
        y=torch.sigmoid(self.conv(y).transpose(-1,-2).unsqueeze(-1))
        return x*y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden=max(channels//reduction,1)
        self.mlp=nn.Sequential(
            nn.Conv2d(channels,hidden,1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden,channels,1,bias=False)
        )
    def forward(self,x):
        a=self.mlp(F.adaptive_avg_pool2d(x,1))
        m=self.mlp(F.adaptive_max_pool2d(x,1))
        return x*torch.sigmoid(a+m)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size,padding=kernel_size//2,bias=False)
    def forward(self,x):
        avg=x.mean(dim=1,keepdim=True)
        mx=x.max(dim=1,keepdim=True).values
        return x*torch.sigmoid(self.conv(torch.cat([avg,mx],dim=1)))

class CBAM(nn.Module):
    def __init__(self,channels,reduction=16,spatial_kernel=7):
        super().__init__()
        self.ca=ChannelAttention(channels,reduction)
        self.sa=SpatialAttention(spatial_kernel)
    def forward(self,x):
        return self.sa(self.ca(x))


