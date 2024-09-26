import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import numbers

from torchvision import models

# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight


# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)

#     def forward(self, x):
#         if len(x.shape)==4:
#             h, w = x.shape[-2:]
#             return to_4d(self.body(to_3d(x)), h, w)
#         else:
#             return self.body(x)

# def to_4d(x, h, w):
#     return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')






# class XCA(nn.Module):
#     """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
#      sum. The weights are obtained from the (softmax normalized) Cross-covariance
#     matrix (Q^T K \\in d_h \\times d_h)
#     """

#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
#         qkv = qkv.permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         q = q.transpose(-2, -1)
#         k = k.transpose(-2, -1)
#         v = v.transpose(-2, -1)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'temperature'}
    

# class PatchUnEmbed(nn.Module):
#     def __init__(self,basefilter) -> None:
#         super().__init__()
#         self.nc = basefilter
#     def forward(self, x,x_size):
#         B,HW,C = x.shape
#         x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
#         return x
# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
#         super().__init__()
#         # patch_size = to_2tuple(patch_size)
#         self.patch_size = patch_size
#         self.flatten = flatten

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
#         self.norm = LayerNorm(embed_dim,'BiasFree')

#     def forward(self, x):
#         #??b,c,h,w)->(b,c*s*p,h//s,w//s)
#         #(b,h*w//s**2,c*s**2)
#         B, C, H, W = x.shape
#         # x = F.unfold(x, self.patch_size, stride=self.patch_size)
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         # x = self.norm(x)
#         return x
    
# class CB2d(nn.Module):
#     def __init__(self, inplanes):
#         super(CB2d, self).__init__()
#         self.inplanes = inplanes
#         # self.planes = inplanes // 2
#         self.channel_mul_conv = nn.Sequential(         
#                 PatchEmbed(in_chans=self.inplanes, embed_dim=self.inplanes,
#                                           patch_size=1,
#                                           stride=1),
                
#                 XCA(self.inplanes),
#             )
#         self.unembed = PatchUnEmbed(self.inplanes)
#         # self.cv1 = nn.Conv2d(in_channels=inplanes, out_channels=self.planes, kernel_size=1)
#         # self.LayerNorm = nn.LayerNorm([self.planes,1,1]),
#         # self.relu = nn.ReLU(inplace=True)
#         # self.cv2 = nn.Conv2d(in_channels=self.planes, out_channels=inplanes, kernel_size=1)
#         self.caozuo = nn.Sequential(
#                 nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1),
#                 nn.BatchNorm2d(num_features=self.inplanes),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1),        
#             )


#     def forward(self, x): #x: torch.Size([1, 32, 64, 64])
#             _, _, h, w = x.shape
#             context = x   #context: torch.Size([1, 32, 64, 64])
            
#             middle = self.channel_mul_conv(context) #middle: torch.Size([1, 4096, 32])
#             middle = self.unembed(middle,(h,w)) #middle: torch.Size([1, 32, 64, 64])

#             # m = self.cv1(middle) #m: torch.Size([1, 16, 64, 64])

#             # m = self.LayerNorm(m)
#             # print("m:",m.shape)
#             # m = self.relu(m)
#             # m = self.cv2(m)
#             m = self.caozuo(middle)

#             out = m*context   
#             return out














# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = CB2d(dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x = torch.rand(1, dim, 64, 64)


# 通过 DLK 模块进行前向传播
output = dlk_module(x)

# 打印输出形状
print("Output shape:", output.shape)