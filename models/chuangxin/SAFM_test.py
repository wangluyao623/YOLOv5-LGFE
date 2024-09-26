# YOLOv5  by Ultralytics, AGPL-3.0 license
"""Common modules."""


import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F


from torchvision import models


# Spatially-Adaptive Feature Modulation
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Multiscale feature representation
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim, bias=False) for i in range(self.n_levels)])

        # Feature aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**(i+1), w//2**(i+1))
                # p_size = (max(h // 2**(i+1), 1), max(w // 2**(i+1), 1))
                # p_size = (max(h // 2**(i+1), 2), max(w // 2**(i+1), 2))

                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        # Feature modulation
        out = self.act(out) * x
        return out


# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = SAFM(dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x = torch.rand(1, dim, 64, 64)


# 通过 DLK 模块进行前向传播
output = dlk_module(x)

# 打印输出形状
print("Output shape:", output.shape)


