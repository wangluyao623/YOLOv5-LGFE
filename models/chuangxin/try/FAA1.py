import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 FAA 模块
class FAA(nn.Module):
    # Siamese features assimilating assistant module
    def __init__(self,in_channels):
        super().__init__()
        # self.in_channels = in_channels
        self.AAFEB1 = AAFEB(in_channels//2, in_channels//2)
        self.AAFEB2 = AAFEB(in_channels//2, in_channels//2)
        
        # self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.cv = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1)
        self.cv1 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
        self.cv2 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
        self.cv3 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
        self.cv4 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        # print("x:",x.shape)
        x1 = x[0] #[1, 32, 64, 64]
        # print("x1:",x1.shape)
        x2 = x[1]
        # print("x2:",x2.shape)
        shortcut = x1
    
        
        chunkx1_1, chunkx1_2 = torch.chunk(x1, chunks=2, dim=1) #[1, 16, 64, 64]

        chunkx1_1 = self.cv1(chunkx1_1) #
        # print("chunkx1_1:",chunkx1_1.shape)
        chunkx1_2 = self.AAFEB1(chunkx1_2)
        x12 = torch.cat((chunkx1_1, chunkx1_2), dim=1)
        x12 = F.relu(x12,inplace=True)


        chunkx1_3, chunkx1_4 = torch.chunk(x12, chunks=2, dim=1)

        chunkx1_3 = self.cv2(chunkx1_3)
        chunkx1_4 = self.AAFEB2(chunkx1_4)
        x1_out = torch.cat((chunkx1_3, chunkx1_4), dim=1)
        x1_out = F.relu(x1_out+shortcut,inplace=True)




        shortcut = x2
        
        chunkx2_1, chunkx2_2 = torch.chunk(x2, chunks=2, dim=1)

        chunkx2_1 = self.cv3(chunkx2_1)
        chunkx2_2 = self.AAFEB1(chunkx2_2)
        x22 = torch.cat((chunkx2_1, chunkx2_2), dim=1)
        x22 = F.relu(x22,inplace=True)


        chunkx2_3, chunkx2_4 = torch.chunk(x22, chunks=2, dim=1)

        chunkx2_3 = self.cv4(chunkx2_3)
        chunkx2_4 = self.AAFEB2(chunkx1_4)
        x2_out = torch.cat((chunkx2_3, chunkx2_4), dim=1)
        x2_out = F.relu(x2_out+shortcut,inplace=True)

        return x2_out

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
    faa_model = FAA(in_channels=32)
    
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