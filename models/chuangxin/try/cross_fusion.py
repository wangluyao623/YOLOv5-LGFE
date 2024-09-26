import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp


from torchvision import models
from mmengine.model import BaseModule
import torch.nn.functional as F
from einops import rearrange



# class Fusion_Unit(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(Fusion_Unit, self).__init__()
#         self.conv_1 = nn.Conv2d(dim_in, dim_out, (3, 3), stride=(1, 1), padding=0)
#         self.BN_1 = nn.BatchNorm2d(dim_out)
#         self.deconv = nn.ConvTranspose2d(dim_out, dim_in, stride=1, kernel_size=3, padding=0, output_padding=0) # output = (input-1)*stride + output_padding - 2padding + kernel_size
#         self.BN_2 = nn.BatchNorm2d(dim_in)
#         self.conv_2 = nn.Conv2d(dim_in, dim_out, (3, 3), stride=(1, 1), padding=0)
#         self.BN_3 = nn.BatchNorm2d(dim_out)

#     def forward(self, F_M):
#         """
#         F_M: [b,c,h,w]
#         """
#         x_sq = F.gelu(self.BN_1(self.conv_1(F_M)))
#         x_ex = F.gelu(self.BN_2(self.deconv(x_sq)))
#         Residual = F_M - x_ex
#         x_r = F.gelu(self.BN_3(self.conv_2(Residual)))
#         x_out = x_sq + x_r

#         return x_out



# class Cross_Fusion(nn.Module):
#     def __init__(self, dim_head, heads, cls):
#         super(Cross_Fusion, self).__init__()
#         self.convH = nn.Conv2d(32, 128, (3, 3), stride=(1, 1), padding=1)
#         self.BN_H1 = nn.BatchNorm2d(128)
#         self.convL = nn.Conv2d(32, 128, (3, 3), stride=(1, 1), padding=1)
#         self.BN_L1 = nn.BatchNorm2d(128)

#         self.num_heads = heads
#         self.dim_head = dim_head
#         self.Hto_q = nn.Linear(128, dim_head * heads, bias=False)
#         self.Hto_k = nn.Linear(128, dim_head * heads, bias=False)
#         self.Hto_v = nn.Linear(128, dim_head * heads, bias=False)
#         self.Lto_q = nn.Linear(128, dim_head * heads, bias=False)
#         self.Lto_k = nn.Linear(128, dim_head * heads, bias=False)
#         self.Lto_v = nn.Linear(128, dim_head * heads, bias=False)
#         self.rescaleH = nn.Parameter(torch.ones(heads, 1, 1))
#         self.rescaleL = nn.Parameter(torch.ones(heads, 1, 1))
#         self.projH = nn.Linear(dim_head * heads, 128, bias=True)
#         self.projL = nn.Linear(dim_head * heads, 128, bias=True)
#         self.LN_H2 = nn.LayerNorm(128)
#         self.LN_L2 = nn.LayerNorm(128)

#         self.FU_1 = Fusion_Unit(256, 360)
#         self.FU_2 = Fusion_Unit(360, 512)
#         self.FU_3 = Fusion_Unit(512, 512)

#         self.GAP = nn.AdaptiveAvgPool2d(1)
#         self.FL = nn.Flatten()
#         self.fcr = nn.Linear(512, cls)

#         self.cv1 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)

#     def forward(self, F_H, F_L):  #1, 32, 64, 64
        
#         """
#         F_H,F_L: [b,c,h,w]
#         """
      

    
#         #feature embedding
#         F_H = F.relu(self.BN_H1(self.convH(F_H))) #[1, 128, 64, 64]
#         F_L = F.relu(self.BN_L1(self.convH(F_L)))


#         #stage 1 for feature cross
#         b, c, h, w = F_H.shape
#         F_H = F_H.permute(0, 2, 3, 1) #1, 64, 64, 128
#         F_L = F_L.permute(0, 2, 3, 1)

        

#         F_H = F_H.reshape(b, h * w, c) #[1, 4096, 128]
#         F_L = F_L.reshape(b, h * w, c)

        


#         Hq_inp = self.Hto_q(F_H)
#         Hk_inp = self.Hto_k(F_H)
#         Hv_inp = self.Hto_v(F_H)
#         Hq, Hk, Hv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
#                       (Hq_inp, Hk_inp, Hv_inp))  # 对qkv调整形状
#         Lq_inp = self.Lto_q(F_L)
#         Lk_inp = self.Lto_k(F_L)
#         Lv_inp = self.Lto_v(F_L)
#         Lq, Lk, Lv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
#                       (Lq_inp, Lk_inp, Lv_inp))

#         Hq = F.normalize(Hq, dim=-2, p=2)
#         Hk = F.normalize(Hk, dim=-2, p=2)
#         Lq = F.normalize(Lq, dim=-2, p=2)
#         Lk = F.normalize(Lk, dim=-2, p=2)

#         attnH = (Hk.transpose(-2, -1) @ Lq)
#         attnH = attnH * self.rescaleH
#         attnH = attnH.softmax(dim=-1)
#         attnL = (Lk.transpose(-2, -1) @ Hq)
#         attnL = attnL * self.rescaleL
#         attnL = attnL.softmax(dim=-1)

#         x_H = Hv @ attnH  # x_H:b,heads,hw,d
#         x_L = Lv @ attnL

#         x_H = x_H.permute(0, 2, 1, 3)  # x_H:b,hw,heads,d
#         x_H = x_H.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_H = self.projH(x_H)  # out_H:b,hw,c
#         x_L = x_L.permute(0, 2, 1, 3)
#         x_L = x_L.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_L = self.projL(x_L)

#         F_H = F_H + out_H
#         F_L = F_L + out_L

#         F_H = F_H.reshape(b, h, w, c)
#         F_H = self.LN_H2(F_H)
#         F_H = F_H.permute(0, 3, 1, 2) # F_H:b,c,h,w
#         F_L = F_L.reshape(b, h, w, c)
#         F_L = self.LN_L2(F_L)
#         F_L = F_L.permute(0, 3, 1, 2)

#         F_M = torch.cat([F_H, F_L], axis=1)
#         F_M = self.cv1(F_M)

#         return F_M














#---------------------------------------------------------------------------------------






# class Cross_Fusion(nn.Module):
#     def __init__(self, dim_head, heads, cls):
#         super(Cross_Fusion, self).__init__()
#         self.convH = nn.Conv2d(32, 128, (3, 3), stride=(1, 1), padding=1)
#         self.BN_H1 = nn.BatchNorm2d(128)
#         self.convL = nn.Conv2d(32, 128, (3, 3), stride=(1, 1), padding=1)
#         self.BN_L1 = nn.BatchNorm2d(128)

#         self.num_heads = heads
#         self.dim_head = dim_head
#         self.Hto_q = nn.Linear(128, dim_head * heads, bias=False)
#         self.Hto_k = nn.Linear(128, dim_head * heads, bias=False)
#         self.Hto_v = nn.Linear(128, dim_head * heads, bias=False)
#         self.Lto_q = nn.Linear(128, dim_head * heads, bias=False)
#         self.Lto_k = nn.Linear(128, dim_head * heads, bias=False)
#         self.Lto_v = nn.Linear(128, dim_head * heads, bias=False)
#         self.rescaleH = nn.Parameter(torch.ones(heads, 1, 1))
#         self.rescaleL = nn.Parameter(torch.ones(heads, 1, 1))
#         self.projH = nn.Linear(dim_head * heads, 128, bias=True)
#         self.projL = nn.Linear(dim_head * heads, 128, bias=True)
#         self.LN_H2 = nn.LayerNorm(128)
#         self.LN_L2 = nn.LayerNorm(128)

#         self.GAP = nn.AdaptiveAvgPool2d(1)
#         self.FL = nn.Flatten()
#         self.fcr = nn.Linear(512, cls)

#         self.cv1 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)

#     def forward(self, F_H, F_L):  #1, 32, 64, 64
#         """
#         F_H,F_L: [b,c,h,w]
#         """
#         #feature embedding
#         F_H = F.relu(self.BN_H1(self.convH(F_H))) #[1, 128, 64, 64]
#         F_L = F.relu(self.BN_L1(self.convH(F_L)))

#         #stage 1 for feature cross
#         b, c, h, w = F_H.shape
#         F_H = F_H.permute(0, 2, 3, 1) #1, 64, 64, 128
#         F_L = F_L.permute(0, 2, 3, 1)

#         F_H = F_H.reshape(b, h * w, c) #[1, 4096, 128]
#         F_L = F_L.reshape(b, h * w, c)

    
#         Hq_inp = self.Hto_q(F_H)
#         Hk_inp = self.Hto_k(F_H)
#         Hv_inp = self.Hto_v(F_H)
#         Hq, Hk, Hv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
#                       (Hq_inp, Hk_inp, Hv_inp))  # 对qkv调整形状
#         Lq_inp = self.Lto_q(F_L)
#         Lk_inp = self.Lto_k(F_L)
#         Lv_inp = self.Lto_v(F_L)
#         Lq, Lk, Lv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
#                       (Lq_inp, Lk_inp, Lv_inp))

#         Hq = F.normalize(Hq, dim=-2, p=2)
#         Hk = F.normalize(Hk, dim=-2, p=2)
#         Lq = F.normalize(Lq, dim=-2, p=2)
#         Lk = F.normalize(Lk, dim=-2, p=2)

#         attnH = (Hk.transpose(-2, -1) @ Lq)
#         attnH = attnH * self.rescaleH
#         attnH = attnH.softmax(dim=-1)
#         attnL = (Lk.transpose(-2, -1) @ Hq)
#         attnL = attnL * self.rescaleL
#         attnL = attnL.softmax(dim=-1)

#         x_H = Hv @ attnH  # x_H:b,heads,hw,d
#         x_L = Lv @ attnL

#         x_H = x_H.permute(0, 2, 1, 3)  # x_H:b,hw,heads,d
#         x_H = x_H.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_H = self.projH(x_H)  # out_H:b,hw,c
#         x_L = x_L.permute(0, 2, 1, 3)
#         x_L = x_L.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_L = self.projL(x_L)

#         F_H = F_H + out_H
#         F_L = F_L + out_L

#         F_H = F_H.reshape(b, h, w, c)
#         F_H = self.LN_H2(F_H)
#         F_H = F_H.permute(0, 3, 1, 2) # F_H:b,c,h,w
#         F_L = F_L.reshape(b, h, w, c)
#         F_L = self.LN_L2(F_L)
#         F_L = F_L.permute(0, 3, 1, 2)

#         F_M = torch.cat([F_H, F_L], axis=1)
#         F_M = self.cv1(F_M)

#         return F_M













# #---------------------------------------------------------------------------------


# class Cross_Fusion(nn.Module):
#     def __init__(self, dim_head, heads, dim):
#         super(Cross_Fusion, self).__init__()
#         # # self.convH = nn.Conv2d(32, 128, (3, 3), stride=(1, 1), padding=1)
#         # self.convH = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        
#         # self.BN_H1 = nn.BatchNorm2d(dim)
#         # # self.convL = nn.Conv2d(32, 128, (3, 3), stride=(1, 1), padding=1)
#         # self.convL = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
#         # self.BN_L1 = nn.BatchNorm2d(dim)

#         self.num_heads = heads
#         self.dim_head = dim_head
#         self.Hto_q = nn.Linear(dim, dim_head * heads, bias=False)
#         self.Hto_k = nn.Linear(dim, dim_head * heads, bias=False)
#         self.Hto_v = nn.Linear(dim, dim_head * heads, bias=False)
#         self.Lto_q = nn.Linear(dim, dim_head * heads, bias=False)
#         self.Lto_k = nn.Linear(dim, dim_head * heads, bias=False)
#         self.Lto_v = nn.Linear(dim, dim_head * heads, bias=False)
#         self.rescaleH = nn.Parameter(torch.ones(heads, 1, 1))
#         self.rescaleL = nn.Parameter(torch.ones(heads, 1, 1))
#         self.projH = nn.Linear(dim_head * heads, dim, bias=True)
#         self.projL = nn.Linear(dim_head * heads, dim, bias=True)
#         self.LN_H2 = nn.LayerNorm(dim)
#         self.LN_L2 = nn.LayerNorm(dim)

#         self.cv1 = nn.Conv2d(2*dim, dim, kernel_size=1, stride=1, padding=0)

#     def forward(self, F_H, F_L):  #1, 32, 64, 64
#         """
#         F_H,F_L: [b,c,h,w]
#         """
#         # #feature embedding
#         # F_H = F.relu(self.BN_H1(self.convH(F_H))) #[1, 128, 64, 64]
#         # F_L = F.relu(self.BN_L1(self.convH(F_L)))

#         #stage 1 for feature cross
#         b, c, h, w = F_H.shape
#         F_H = F_H.permute(0, 2, 3, 1) #1, 64, 64, 128
#         F_L = F_L.permute(0, 2, 3, 1)

#         F_H = F_H.reshape(b, h * w, c) #[1, 4096, 128]
#         F_L = F_L.reshape(b, h * w, c)

    
#         Hq_inp = self.Hto_q(F_H)
#         Hk_inp = self.Hto_k(F_H)
#         Hv_inp = self.Hto_v(F_H)
#         Hq, Hk, Hv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
#                       (Hq_inp, Hk_inp, Hv_inp))  # 对qkv调整形状
#         Lq_inp = self.Lto_q(F_L)
#         Lk_inp = self.Lto_k(F_L)
#         Lv_inp = self.Lto_v(F_L)
#         Lq, Lk, Lv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
#                       (Lq_inp, Lk_inp, Lv_inp))

#         Hq = F.normalize(Hq, dim=-2, p=2)
#         Hk = F.normalize(Hk, dim=-2, p=2)
#         Lq = F.normalize(Lq, dim=-2, p=2)
#         Lk = F.normalize(Lk, dim=-2, p=2)

#         attnH = (Hk.transpose(-2, -1) @ Lq)
#         attnH = attnH * self.rescaleH
#         attnH = attnH.softmax(dim=-1)
#         attnL = (Lk.transpose(-2, -1) @ Hq)
#         attnL = attnL * self.rescaleL
#         attnL = attnL.softmax(dim=-1)

#         x_H = Hv @ attnH  # x_H:b,heads,hw,d
#         x_L = Lv @ attnL

#         x_H = x_H.permute(0, 2, 1, 3)  # x_H:b,hw,heads,d
#         x_H = x_H.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_H = self.projH(x_H)  # out_H:b,hw,c
#         x_L = x_L.permute(0, 2, 1, 3)
#         x_L = x_L.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_L = self.projL(x_L)

#         F_H = F_H + out_H
#         F_L = F_L + out_L

#         F_H = F_H.reshape(b, h, w, c)
#         F_H = self.LN_H2(F_H)
#         F_H = F_H.permute(0, 3, 1, 2) # F_H:b,c,h,w
#         F_L = F_L.reshape(b, h, w, c)
#         F_L = self.LN_L2(F_L)
#         F_L = F_L.permute(0, 3, 1, 2)

#         F_M = torch.cat([F_H, F_L], axis=1)
#         F_M = self.cv1(F_M)

        # return F_M



import torch
import torch.nn as nn

class FDAF(BaseModule):
    """Flow Dual-Alignment Fusion Module.

    Args:
        in_channels (int): Input channels of features.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

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
        
        kernel_size = 5
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, groups=in_channels*2),
            nn.InstanceNorm2d(in_channels*2),
            nn.GELU(),
            nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=False),
        )
        self.cv1 = nn.Conv2d(in_channels=2*dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        """Forward function."""

        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) + x2
        x2_feat = self.warp(x2, f2) + x1
    
     
        output = torch.cat([x1_feat, x2_feat], dim=1)

        output = self.cv1(output)
        
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output


# 示例用法
x1 = torch.randn(1, 32, 64, 64)  # 假设输入1是一个形状为 (batch_size, channels, height, width) 的张量
x2 = torch.randn(1, 32, 64, 64)  # 假设输入2是一个形状为 (batch_size, channels, height, width) 的张量

model = FDAF(in_channels=32)
output = model(x1, x2)
print(output.shape)  # 输出张量的形状


