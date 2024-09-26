import math
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# class FEM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
#         #print("inter_planes:",inter_planes)      # 4
#         self.branch0 = nn.Sequential(
#             BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#             # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
#             BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=1, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#             BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#             BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv(in_planes, out_planes, kernel_size=1, stride=stride)
#         )

#         self.ConvLinear = BasicConv(4 * in_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):     #x: torch.Size([1, 32, 64, 64])
#         # x0 = self.branch0(x)  # x0: torch.Size([1, 32, 64, 64])
#         # x1 = self.branch1(x) # x1: torch.Size([1, 32, 64, 64])
#         # x2 = self.branch2(x) # x2: torch.Size([1, 32, 64, 64])

#         # out = torch.cat((x0, x1, x2), 1) # out1: torch.Size([1, 96, 64, 64])
#         # out = self.ConvLinear(out) #out2: torch.Size([1, 32, 64, 64])
#         # short = self.shortcut(x) #short: torch.Size([1, 32, 64, 64])
#         # out = out * self.scale + short #out3: torch.Size([1, 32, 64, 64])
#         # out = self.relu(out) #out4: torch.Size([1, 32, 64, 64])

#         x0 = self.branch0(x)
#         x1 = self.branch1(x0 + x)
#         x2 = self.branch2(x0 + x1 + x)
#         x3 = self.branch3(x0 + x1 + x2 + x)
#         out = torch.cat((x0, x1, x2,x3), 1)
#         out = self.ConvLinear(out)
#         out = self.relu(out)

#         return out






    

# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
    


# 自定义 GroupBatchnorm2d 类，实现分组批量归一化
class GroupBatchnorm2d(nn.Module):
    def __init__(self, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        self.group_num = group_num
        self.eps = eps
        self.gamma = None
        self.beta = None

    def forward(self, x): 
        N, C, H, W = x.size()
        
        if self.gamma is None or self.gamma.size(0) != C:
            self.gamma = nn.Parameter(torch.randn(C, 1, 1).to(x.device))
            self.beta = nn.Parameter(torch.zeros(C, 1, 1).to(x.device))
        
        # 确保 group_num 不超过通道数 C
        group_num = min(self.group_num, C)
        assert C % group_num == 0, "C must be divisible by group_num"
        
        x = x.view(N, group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        
        return x * self.gamma + self.beta



# x: torch.Size([1, 128, 32, 32])
# self.gamma: torch.Size([128, 1, 1])
# self.beta: torch.Size([128, 1, 1])
# x: torch.Size([1, 256, 16, 16])
# self.gamma: torch.Size([128, 1, 1])
# self.beta: torch.Size([128, 1, 1])












# 自定义 SRU（Spatial and Reconstruct Unit）类
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,  # 输出通道数
                 group_num: int = 16,  # 分组数，默认为16
                 gate_treshold: float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn: bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数

        # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        if torch_gn:
            self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
        else:
            self.gn = GroupBatchnorm2d(group_num=group_num)
        
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigomid = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x = self.gn(x)  # 应用分组批量归一化
        # 动态计算 w_gamma
        if hasattr(self.gn, 'gamma'):
            w_gamma = self.gn.gamma / torch.sum(self.gn.gamma)
        else:
            w_gamma = torch.ones_like(gn_x)
        reweights = self.sigomid(gn_x * w_gamma)  # 计算重要性权重

        # 门控机制
        info_mask = reweights >= self.gate_treshold  # 计算信息门控掩码
        noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
        x_1 = info_mask * x  # 使用信息门控掩码
        x_2 = noninfo_mask * x  # 使用非信息门控掩码
        x = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接

class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super(CRU, self).__init__()

        self.up_channel = up_channel = int(alpha * op_channel)  # 计算上层通道数
        self.low_channel = low_channel = op_channel - up_channel  # 计算下层通道数
        
        # 确保分割的通道数总和等于输入张量的通道数
        assert self.up_channel + self.low_channel == op_channel, \
            "up_channel 和 low_channel 之和必须等于 op_channel"

        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        # 上层特征转换
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)

        # 下层特征转换
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2



# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class ScConv(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()  # 调用父类构造函数

        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size,
                       group_kernel_size=group_kernel_size)  # 创建 CRU 层

    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return x




class MyHeadBlock(nn.Module):
    def __init__(self,in_ch):
        super(MyHeadBlock, self).__init__()
        self.msdconv_ssfc = ScConv(in_ch,in_ch)

    def forward(self,x):
        outs = []
        for level in range(3):
            x[level] = self.msdconv_ssfc(x[level])
            outs.append(x[level])
        return outs
    














    



# # 测试用例
# def test_msdconv_ssfc():
#     x = torch.randn(1, 32, 64, 64)  # Batch of 1, 32 channels, 64x64 size
#     model = MyHeadBlock(in_ch=32)
#     output = model(x)
#     print("output:", output.shape)

# # 运行测试用例
# test_msdconv_ssfc()



