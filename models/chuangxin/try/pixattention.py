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
        pattn2 = self.sigmoid(pattn2) #[1, 32, 64, 64]
        return pattn2
    

# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = PixelAttention(dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x1 = torch.rand(1, dim, 64, 64)
x2 = torch.rand(1, dim, 64, 64)
pattn1 = torch.rand(1, dim, 64, 64)

# 通过 DLK 模块进行前向传播
output = dlk_module(x1,x2,pattn1)

# 打印输出形状
print("Output shape:", output.shape)