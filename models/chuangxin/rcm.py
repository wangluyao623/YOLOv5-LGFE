import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp


from torchvision import models


class RCA(nn.Module):
    def __init__(self, inp,  kernel_size=1, ratio=1, band_kernel_size=11,dw_size=(1,1), padding=(0,0), stride=1, square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, 11, padding=5, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc=inp//ratio
        self.excite = nn.Sequential(
                nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc),
                nn.Sigmoid()
            )
    
    def sge(self, x):
        #[N, D, C, 1]
        x_h = self.pool_h(x)
        print(x_h.shape)
        x_w = self.pool_w(x)
        print(x_w.shape)
        x_gather = x_h + x_w #.repeat(1,1,1,x_w.shape[-1])
        print(x_gather.shape)
        ge = self.excite(x_gather) # [N, 1, C, 1]
        print(ge.shape)
        
        return ge

    def forward(self, x):
        loc=self.dwconv_hw(x) #1,32,64,64
        att=self.sge(x) #1,32,64,64
        out = att*loc
        
        return out
    


# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = RCA(dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x = torch.rand(1, dim, 64, 64)


# 通过 DLK 模块进行前向传播
output = dlk_module(x)

# 打印输出形状
print("Output shape:", output.shape)