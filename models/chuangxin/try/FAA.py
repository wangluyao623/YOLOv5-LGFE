import torch
import torch.nn as nn
# 定义 FAA 模块
# class FAA(nn.Module):
#     # Siamese features assimilating assistant module
#     def __init__(self,in_channels,mid_channels=[64,64,64,64], kernels=7):
#         super().__init__()
#         # self.in_channels = in_channels
#         self.branch1 = FEBranch(in_channels, mid_channels, kernels)
#         self.cv1 = nn.Conv2d(64, in_channels, kernel_size=1)
#         # self.cv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

#     def forward(self,x):
#         x1 = x[0]
#         x2 = x[1]
#         y1 = self.branch1(x1)
#         y2 = self.branch1(x2)
#         # res = []
#         # for i, j in zip(y1, y2):
#         #     z = i + j
#         #     res.append(z)
#         # return res
#         z = y1[-1] + y2[-1]
#         z = self.cv1(z)
#         return z

# # 定义 FEBranch 模块
# class FEBranch(nn.Module):
#     def __init__(self, in_channels, mid_channels=[16,32,64,64], kernels=7):
#         super(FEBranch, self).__init__()
#         self.layers = nn.ModuleList()
#         for c in mid_channels:
#             self.layers.append(FEAA(in_channels, c, kernels))
#             in_channels = c

#     def forward(self, x):
#         y = x
#         res = []
#         for layer in self.layers:
#             y = layer(y)
#             res.append(y)
#         return res

# 定义 FEAA 模块
class FEAA(nn.Module):
    # Assimilating assistant feature extraction
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__()
        # 使用 PyTorch 的层来替换 paddleseg 的 ConvBNReLU
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # CBLKB ------------------------------------------------------
        self.c11 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gck = nn.Conv2d(out_channels, out_channels, kernel_size=kernels, padding=kernels // 2, groups=out_channels)
        
        self.bnr = nn.Sequential(nn.BatchNorm2d(out_channels), nn.GELU())
        self.lastcbr = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
     
    def forward(self, x):
        z = self.cbr1(x)
        z1 = self.c11(z)
        z2 = self.gck(z)
        y = z1 + z2
        return self.lastcbr(self.bnr(y))
    

# 测试 FAA 模块
def test_faa():
    # 实例化模型
    faa_model = FEAA(in_channels=32, mid_channels=[64,64,64,64], kernels=7)
    
    # 创建两个模拟输入张量，形状为 (batch_size, channels, height, width)
    x1 = torch.randn(1, 32, 64, 64)  # 模拟第一个输入图像
    x2 = torch.randn(1, 32, 64, 64)  # 模拟第二个输入图像
    
    # 将两个输入放入列表
    inputs = [x1, x2]
    
    # 执行前向传播
    output = faa_model(inputs)
    
    # 打印输出特征图的形状
    print(f"Output shape: {output.shape}")

# 运行测试
test_faa()