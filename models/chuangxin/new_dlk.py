import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp


from torchvision import models

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
        # self.spatial_se = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
        #     nn.Sigmoid()
        # )
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
        print(x_h.shape)
        x_w = self.pool_w(x)
        print(x_w.shape)
        x_h, x_w = self.Bi_GRU(x_h, x_w)
        x_gather = x_h + x_w
        ge = self.excite(x_gather)
        # ge = self.cv3(ge) #c-2c

        # ge1 = ge[:, :ge.size(1) // 2, :, :]  # 取前一半通道
        # ge2 = ge[:, ge.size(1) // 2:, :, :]  # 取后一半通道

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

# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = DLK(dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x1 = torch.rand(1, dim, 64, 64)
x2 = torch.rand(1, dim, 64, 64)

# 通过 DLK 模块进行前向传播
output = dlk_module(x1,x2)

# 打印输出形状
print("Output shape:", output.shape)