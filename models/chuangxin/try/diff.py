import torch
import torch.nn as nn
import torch.nn.functional as F

# # 定义 FAA 模块
# class FAA(nn.Module):
#     # Siamese features assimilating assistant module
#     def __init__(self,in_channels,kernels=7):
#         super().__init__()
#         # self.in_channels = in_channels
#         self.AAFEB1 = AAFEB(in_channels, in_channels)
#         self.AAFEB2 = AAFEB(in_channels, in_channels)
        
#         # self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         # self.cv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.diff1 = DifferenceFeatureModule(in_channels)
#         self.diff2 = DifferenceFeatureModule(in_channels)
#         self.diff3 = DifferenceFeatureModule(in_channels)

#     def forward(self,x):
#         x1 = x[0]
#         x2 = x[1]
#         diff1 = self.diff1(x1,x2)
#         y1_1 = self.AAFEB1(x1)
#         y2_1 = self.AAFEB1(x2)
#         diff2 = self.diff1(y1_1,y2_1)
#         y1_2 = self.AAFEB2(y1_1)
#         y2_2 = self.AAFEB2(y2_1)
#         diff3 = self.diff1(y1_2,y2_2)
        
#         z = y1_2 + y2_2 + diff1 + diff2 +diff3
#         # z = self.cv1(z)
#         return z


# class AAFEB(nn.Module):
#     def __init__(self, in_channels, out_channels, kernels=7):
#         super().__init__()
#         # 使用 PyTorch 的层来替换 paddleseg 的 ConvBNReLU
#         self.cbr1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         # CBLKB ------------------------------------------------------
#         self.c11 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
#         self.gck = nn.Conv2d(out_channels, out_channels, kernel_size=kernels, padding=kernels // 2, groups=out_channels)
        
#         self.bnr = nn.Sequential(nn.BatchNorm2d(out_channels), nn.GELU())
#         self.lastcbr = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
     
#     def forward(self, x):
#         z = self.cbr1(x)
#         z1 = self.c11(z)
#         z2 = self.gck(z)
#         y = z1 + z2
#         return self.lastcbr(self.bnr(y))



# class DifferenceFeatureModule(nn.Module):
#     def __init__(self, in_channels):
#         super(DifferenceFeatureModule, self).__init__()
        
#         # 1x1卷积层
#         self.conv1x1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.conv1x1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
#         # 3x3卷积层
#         self.conv3x3 = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1)
        
#         # 平均池化层
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # 最终3x3卷积
#         self.final_conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
#     def forward(self, f1, f2):
#         # 对 f1 和 f2 进行逐元素加法与减法
#         add_result = f1 + f2
#         sub_result = f1 - f2
#         f = torch.cat((add_result, sub_result), dim=1)

#         ff = self.conv3x3(f)
        
        
#         fff = self.avgpool(ff)
        
#         # 对池化后的结果进行1x1卷积
#         fff1 = self.conv1x1_2(fff)

#         fff2 = self.conv1x1_1(fff)
        
#         # 将上述两个结果进行逐元素相乘
#         out1 = fff1 * f1
#         out2 = fff2 * f2
        
#         # 将相乘结果与加法结果相加
#         final_sum = out1 + out2
        
#         # 通过最终的3x3卷积得到输出
#         output = self.final_conv3x3(final_sum)
        
#         return output

class AAFEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用 PyTorch 的层来替换 paddleseg 的 ConvBNReLU
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # CBLKB ------------------------------------------------------
        self.c11 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gck = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        
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
    faa_model = AAFEB(in_channels=32,out_channels=32)
    
    # 创建两个模拟输入张量，形状为 (batch_size, channels, height, width)
    x1 = torch.randn(1, 32, 64, 64)  # 模拟第一个输入图像
   
    
    # 将两个输入放入列表
    inputs = x1
    
    # 执行前向传播
    output = faa_model(inputs)
    
    # 打印输出特征图的形状
    print(f"Output shape: {output.shape}")

# 运行测试
test_faa()