#

import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp


from torchvision import models
from einops.layers.torch import Rearrange



class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(3 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, pattn1):  #x [1, 32, 64, 64]
        B, C, H, W = x1.shape
        x1 = x1.unsqueeze(dim=2)  # B, C, 1, H, W #[1, 32, 1, 64, 64]
        x2 = x2.unsqueeze(dim=2)  # B, C, 1, H, W #[1, 32, 1, 64, 64]
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W [1, 32, 1, 64, 64]
        f = torch.cat([x1, x2, pattn1], dim=2)  # B, C, 2, H, W [1, 32, 2, 64, 64]
        f = Rearrange('b c t h w -> b (c t) h w')(f) # [1, 64, 64, 64]
        pattn2 = self.pa2(f)  #[1, 32, 64, 64]
        # pattn2 = self.sigmoid(pattn2) #[1, 32, 64, 64]
        return pattn2


class FEM(nn.Module):  #特征增强模块
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        #print("inter_planes:",inter_planes)      # 4
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
            BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
            BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
            BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, out_planes, kernel_size=1, stride=stride)
        )

        self.ConvLinear = BasicConv(4 * in_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):     #x: torch.Size([1, 32, 64, 64])
        # x0 = self.branch0(x)  # x0: torch.Size([1, 32, 64, 64])
        # x1 = self.branch1(x) # x1: torch.Size([1, 32, 64, 64])
        # x2 = self.branch2(x) # x2: torch.Size([1, 32, 64, 64])

        # out = torch.cat((x0, x1, x2), 1) # out1: torch.Size([1, 96, 64, 64])
        # out = self.ConvLinear(out) #out2: torch.Size([1, 32, 64, 64])
        # short = self.shortcut(x) #short: torch.Size([1, 32, 64, 64])
        # out = out * self.scale + short #out3: torch.Size([1, 32, 64, 64])
        # out = self.relu(out) #out4: torch.Size([1, 32, 64, 64])

        x0 = self.branch0(x)
        x1 = self.branch1(x0 + x)
        x2 = self.branch2(x0 + x1 + x)
        x3 = self.branch3(x0 + x1 + x2 + x)
        out = torch.cat((x0, x1, x2,x3), 1)
        out = self.ConvLinear(out)
        out = self.relu(out)

        return out
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



#交互？先暂定，做一个最简单的交互

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''
 
    def __init__(self, in_planes, r=4):
        super(AFF, self).__init__()
       
        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0),
        )
 
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0),
        )
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x, residual):
        xf1 = x + residual

        xf1 = self.local_att(xf1)
        xf1 = self.global_att(xf1)
        wei = self.sigmoid(xf1)
 
        result = x * wei + residual * (1 - wei)
        return result
    


class MyFusionBlock(nn.Module):
 
    def __init__(self, in_planes, r=4):
        super(AFF, self).__init__()
        self.aff1 = AFF(in_planes)
        self.fem1 = FEM(in_planes,in_planes)
        self.aff2 = AFF(in_planes)
        self.fem2 = FEM(in_planes,in_planes)
        self.pa = PixelAttention(in_planes)
        self.cv1 = nn.Conv2d(in_channels=2*dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        
 
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        t0 = self.aff1(x)
        
        t1 = self.fem1(t0)
        t2 = t1+x1
        t3 = self.fem2(t2)
        t4 = t0+t3
        t5 = t0+t4
        t6 = t3+t5

        t7 = self.pa(x1,x2,t6)


        weight = self.sigmoid(t7)
    
        r1= x1 * weight 
        r2 = x2 * (1 - weight)
        result = x1+x2+r1+r2
        return result
    


    

# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = AFF(dim,dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x1 = torch.rand(1, dim, 64, 64)
x2 = torch.rand(1, dim, 64, 64)

# 通过 DLK 模块进行前向传播
output = dlk_module(x1,x2)

# 打印输出形状
print("Output shape:", output.shape)