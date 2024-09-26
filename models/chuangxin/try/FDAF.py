import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp


from torchvision import models
from mmengine.model import BaseModule
import torch.nn.functional as F


class FDAF(BaseModule):

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        conv_cfg=None
        norm_cfg=dict(type='IN')
        act_cfg=dict(type='GELU')
        
        kernel_size = 3
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels*2, 2*in_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, groups=in_channels*2),
            nn.InstanceNorm2d(in_channels),
            nn.GELU(),
            # nn.Conv2d(in_channels, 4, kernel_size=1, padding=0, bias=False),
        )
        self.cv1 = nn.Conv2d(in_channels=2*dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.cv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.cv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid =  nn.Sigmoid()


    def forward(self, x1, x2): #1 32 64 64

        """Forward function."""

        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)

        f1, f2 = torch.chunk(flow, 2, dim=1)

        f1 = self.cv2(f1)

        f1 = self.sigmoid(f1)
        f2 = self.pool(f2)
        f2 = self.cv3(f2)
        f2 = self.sigmoid(f2)


        f1 = f1*x2
        f2 = f2*x1
     
        output = torch.cat([f1, f2], dim=1)

        output = self.cv1(output)
        
        return output


   


# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = FDAF(dim,dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x1 = torch.rand(1, dim, 64, 64)
x2 = torch.rand(1, dim, 64, 64)

# 通过 DLK 模块进行前向传播
output = dlk_module(x1,x2)

# 打印输出形状
print("Output shape:", output.shape)

















