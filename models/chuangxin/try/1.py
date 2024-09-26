import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

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
from einops.layers.torch import Rearrange


from torchvision import models






class TAU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TAU, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, 2*in_channels, kernel_size=1, padding=0, stride=1)
        self.cv2 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, stride=1)
        self.cv3 = nn.Conv2d(2*in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)
       
        
        

    def forward(self, x): 
        x1 = self.relu(self.cv1(x))                
        x2 = self.relu(self.cv2(x1))  
        x3 = self.cv3(x2)
        x4 = x+x3     
        x4 = self.relu(x4)        
        out = x4

        return out
    

class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InvertedBottleneck, self).__init__()
      
        self.tau = TAU(in_channels, out_channels)
        
    def forward(self, x):
        for _ in range(2):
            x = self.tau(x)
        return x



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------XCA--------------------------------------------------------------------------------------------------
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

#---------------------------------------------------------------------------------GC--------------------------------------------------------------------------------------------------
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True
class DynamicLayerNorm(nn.Module):
    def __init__(self, num_features):
        super(DynamicLayerNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return self.weight.view(1, self.num_features, 1, 1) * (x - mean) / (std + 1e-5) + self.bias.view(1, self.num_features, 1, 1)

class CB2d(nn.Module):
    def __init__(self, inplanes, pool='att', fusions=['channel_add', 'channel_mul']):
        super(CB2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = inplanes // 2
        self.pool = pool
        self.fusions = fusions


        self.caozuo = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1),
                nn.BatchNorm2d(num_features=self.inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1),        
            )

        
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                DynamicLayerNorm(self.planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                # nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # nn.LayerNorm([self.planes, 1, 1]),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
                
                PatchEmbed(in_chans=self.inplanes, embed_dim=self.inplanes,
                                          patch_size=1,
                                          stride=1),
                
                XCA(self.inplanes),
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()
        self.unembed = PatchUnEmbed(self.inplanes)

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':  # iscyy
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(3)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        _, _, h, w = x.shape
        if self.channel_mul_conv is not None:
            middle = self.channel_mul_conv(x)
            middle = self.unembed(middle,(h,w))
            # out = x * middle #[1,32,64,64]      


            m = self.caozuo(middle)

            out = x + m
        return out
    


#-----------------------------------------Fusion------------------------------------------------------
class DirectionalBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DirectionalBiGRU, self).__init__()
        self.hidden_dim = hidden_dim

        # 用于处理横向（x轴）特征的双向GRU层
        self.bi_gru_x = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # 用于处理纵向（y轴）特征的双向GRU层
        self.bi_gru_y = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)

        # 1x1卷积层用于调整通道数
        self.conv1x1_x = nn.Conv2d(in_channels=2*hidden_dim, out_channels=input_dim, kernel_size=1)
        self.conv1x1_y = nn.Conv2d(in_channels=2*hidden_dim, out_channels=input_dim, kernel_size=1)

        
    def forward(self, x, y):#1*32*64*1   1*32*1*64

        y = y.squeeze(2)  #1*32*64
        x = x.squeeze(3)  #1*32*64

        # 转置得到 (batch_size, W, C) 和 (batch_size, H, C)
        y = y.transpose(1, 2)  # 1*64*32
        x = x.transpose(1, 2)  # 1*64*32

        # 通过双向GRU层
        output_x, _ = self.bi_gru_x(x)  # 输出形状: (batch_size, W, 2*hidden_dim) 1*64*64
        output_y, _ = self.bi_gru_y(y)  # 输出形状: (batch_size, H, 2*hidden_dim) 1*64*64

        # 转置回原始形状
        output_x = output_x.transpose(1, 2)  # (batch_size, 2*hidden_dim, W) #1,64,64
        output_y = output_y.transpose(1, 2)  # (batch_size, 2*hidden_dim, H) #1,64,64

        # 增加额外维度以匹配原始输入形状
        output_y = output_y.unsqueeze(2)  # (batch_size, 2*hidden_dim, 1, W) 1*64*64*1
        output_x = output_x.unsqueeze(3)  # (batch_size, 2*hidden_dim, H, 1) 1*64*1*64

        # 使用1x1卷积调整通道数
        output_x = self.conv1x1_x(output_x)  # (batch_size, input_dim, 1, W) 1*32*64*1
        output_y = self.conv1x1_y(output_y)  # (batch_size, input_dim, H, 1) 1*32*1*64

        return output_x, output_y
    


class DLK(nn.Module):
    def __init__(self, dim,ratio=1,m = -0.50):
        super().__init__()
        self.sigmoid =  nn.Sigmoid()
        # self.fc1 = nn.Linear(dim, dim)
        self.cv1 = nn.Conv2d(in_channels=2*dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.cv2_1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.cv2_2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.cv3 = nn.Conv2d(in_channels=dim, out_channels=2*dim, kernel_size=3, stride=1, padding=1)
        self.cv4_1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.cv4_2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

        self.dwconv_hw = nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        gc=dim//ratio
        self.excite = nn.Sequential(
                nn.Conv2d(dim, gc, kernel_size=(1, 11), padding=(0, 5), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, dim, kernel_size=(11, 1), padding=(5, 0), groups=gc),
                # nn.Sigmoid()
            )
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.Bi_GRU = DirectionalBiGRU(dim,dim)

    def forward(self,x1,x2): #1,32,64,64||1,32,64,64  
    #----------------------------------------------------------------------------
        att = torch.cat([x1, x2], dim=1) #1,64,64,64
        x = self.cv1(att)  #2c-c
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_h, x_w = self.Bi_GRU(x_h, x_w)
        x_gather = x_h + x_w
        ge = self.excite(x_gather)

        ge1 = self.cv4_1(ge)
        ge2 = self.cv4_2(ge)

        ge1 = self.sigmoid(ge1)
        ge2 = self.sigmoid(ge2)

        out1 = x1 * ge1
        out2 = x2 * ge2

        out1 = self.cv2_1(out1 + x1)
        out2 = self.cv2_2(out2 + x2)

        mix_factor = self.sigmoid(self.w)
        out1 = out1 * mix_factor.expand_as(out1)
        out2 = out2 * (1 - mix_factor.expand_as(out2))
        out = out1 + out2
        return out

#------------------------------------------------------------------------------------------------------------------------------------------

class CoDEM2(nn.Module):

    def __init__(self,channel_dim):
        super(CoDEM2, self).__init__()

        self.channel_dim=channel_dim
        
        self.tau = TAU(channel_dim,channel_dim)
        self.local = InvertedBottleneck(channel_dim,channel_dim)
        self.cb2d = CB2d(channel_dim)
        self.dlk = DLK(channel_dim)

    def forward(self,x): #1,32,64,64
        x1 = self.local(x) #1 32 64 64
        x2 = self.cb2d(x)#1 32 64 64
        out = self.dlk(x1,x2) 
        
        return out


import torch.nn.functional as F

class DSCBlock(nn.Module):
    def __init__(self, c1,c2,shortcut=True,g=1,e=1,ratio=16, kernel_size=7):
        super(DSCBlock, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1,g=g)
        self.add = shortcut and c1 == c2

        self.dem = CoDEM2(c2)


    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        out = self.dem(x2)

        out = F.interpolate(out, size=x1.size()[2:], mode='bilinear', align_corners=False)

        
        return x + out if self.add else out


# class C3DSC(C3):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
#         convolutions, and expansion.
#         """
#         super().__init__(c1,c2,n,shortcut,g,e)
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.Sequential(*(DSCBottleNeck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

#     def forward(self, x):
#         """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))



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
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #??b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    

# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = DSCBlock(dim,dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x = torch.rand(1, dim, 64, 64)


# 通过 DLK 模块进行前向传播
output = dlk_module(x)

# 打印输出形状
print("Output shape:", output.shape)