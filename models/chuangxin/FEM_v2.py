# YOLOv5  by Ultralytics, AGPL-3.0 license
"""Common modules."""


import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp


from torchvision import models


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.silu = nn.SiLU()
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, in_planes//4, kernel_size=1, stride=1)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, in_planes//4, kernel_size=3, stride=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1),
            BasicConv((inter_planes // 2) * 3, in_planes//4, kernel_size=5, stride=1)
            
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=5, stride=1),
            BasicConv(2 * inter_planes, inter_planes//4, kernel_size=7, stride=1)
        )
        

    def forward(self, x):
        x = self.silu(x)
        print("x",x.shape) #x torch.Size([1, 32, 64, 64])
        x0 = self.branch0(x) #x0 torch.Size([1, 8, 64, 64])
        print("x0",x0.shape) #x1 torch.Size([1, 8, 62, 62])
        x1 = self.branch1(x) #x2 torch.Size([1, 8, 58, 58])
        print("x1",x1.shape) #x3 torch.Size([1, 1, 52, 52])
        x2 = self.branch2(x)
        print("x2",x2.shape)
        x3 = self.branch3(x)
        print("x3",x3.shape)

        out = torch.cat((x0, x1, x2,x3), 1)
        out = out + x
        

        return out
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.silu = nn.SiLU()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.silu(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        
        return x
    

# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = FEM(dim,dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x = torch.rand(1, dim, 64, 64)


# 通过 DLK 模块进行前向传播
output = dlk_module(x)

# 打印输出形状
print("Output shape:", output.shape)