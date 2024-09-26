

# class DSCBottleNeck(nn.Module):
#     def __init__(self, c1,c2,shortcut=True,g=1,e=0.5,ratio=16, kernel_size=7):
#         super(DSCBottleNeck, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1,g=g)
#         self.add = shortcut and c1 == c2

#         self.dsconv1 = CoordConv(c2, c2, 1, 1)
#         self.dsconv2 = CoordConv(c2, c2, 3, 1)
#         self.dsconv3 = CoordConv(c2, c2, 5, 1)
#         self.conv1 = Conv(c2,2*c_,3,1)
#         self.conv2 = Conv(2*c_,c2,3,1)
       

#     def forward(self, x):
#         x2 = self.cv2(self.cv1(x))
#         out1 = self.dsconv1(x2)
#         out1_1  = self.conv1(out1)
#         out1_2 = self.conv2(out1_1)


#         out2 = self.dsconv2(x2)
#         out2_2 = out1_2 * out2
#         out2_3 = out2_2+out1_2
#         out2_4 = self.conv1(out2_3)
#         out2_5 = self.conv2(out2_4)


#         out3 = self.dsconv3(x2)
#         out3_2 = out2_5 * out3
#         out3_3 = out3_2 + out2_5
#         out = out3_3
        
#         return x + out if self.add else out



import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
from einops import rearrange
import numbers

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
#         B,C,H,W = x.shape
#         # 将输入数据重塑为 B, H*W, 3, num_heads, C//num_heads
#         qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, C // self.num_heads)

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
#         x = (attn @ v).transpose(1, 2).reshape(B, H, W, C // self.num_heads * self.num_heads)
#         print("x:",x.shape) #x: torch.Size([1, 64, 64, 32])
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
        


#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'temperature'}
    

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

#     def forward(self, x): #1*32*64*64
#         B,C,H,W = x.size()
#         x=x.view(B,H*W,C) #[1, 4096, 32]
#         # print(x.shape)   
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
#         B,N,C = x.size()
#         x = x.view(B,C,H,W)
#         return x

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'temperature'}

# class PatchEmbed(nn.Module):
#     def __init__(self, img_size=64,  # 输入图片大小
# 			    patch_size=4,  # 分块大小
# 			    in_c=32,  # 输入图片的通道数
# 			    embed_dim=512,  # 经过PatchEmbed后的分块的通道数
# 			    norm_layer=None): # 标准化层
#         super(PatchEmbed, self).__init__()
#         img_size = (img_size, img_size)  #将img_size、patch_size转为元组
#         patch_size = (patch_size, patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         # // 是一种特殊除号，作用为向下取整
#         # grid_size:分块后的网格大小，即一张图片切分为块后形成的网格结构，理解不了不用理解，就是为了求出分块数目的
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]  # 分块数量
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  # 分块用的卷积
#         # 如果norm_layer为None，就使用一个空占位层，就是看要不要进行一个标准化
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#         # nn.Identity()层是用来占位的，没什么用

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # assert是python的断言，当后面跟的是False时就会停下
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
#         """
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         1.第一步将x做卷积 [B, 3, 224, 224] -> [B, 768, 14, 14]
#         2.从位序为2的维度开始将x展平 [B, 768, 14, 14] -> [B, 768, 196]
#         3.转置[B, 196, 768] 得到batch批次，每个批次有196个“词”，每个“词”有768维
#         """
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x

# class PatchUnEmbed(nn.Module):
#     r""" Image to Patch Unembedding
#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """
 
#     def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]
 
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
 
#     def forward(self, x, x_size):
#         B, HW, C = x.shape
#         x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
#         print(x.shape)
#         return x
 
#     def flops(self):
#         flops = 0
#         return flops

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class PatchUnEmbed(nn.Module):
    def __init__(self, basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, stride=4, in_chans=36, embed_dim=32 * 32 * 32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm2d(embed_dim, 'BiasFree')

    def forward(self, x):
        # （b,c,h,w)->(b,c*s*p,h//s,w//s)
        # (b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x





class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}










# 创建 XCA 模块实例
# model = XCA(dim=32, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)

model = PatchEmbed(img_size = 64,patch_size = 4,in_c = 32,embed_dim = 512,)

# 准备输入数据
input_data = torch.randn(1, 32, 64, 64)  # 输入数据维度为 1*32*64*64




# 模型推理
output1 = model(input_data)

# 打印输出结果的形状
print("Output1 shape:", output1.shape)


model1 = XCA(dim=512, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1)

# 通过 XCA 模块进行前向传播
output2 = model1(output1)
print("Output2 shape:", output2.shape)