import torch
import torch.nn as nn

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

# 示例使用
batch_size = 1
C = 32
H = 64
W = 64
hidden_dim = 32

# 示例输入特征图
x = torch.randn(batch_size, C, H, 1)
y = torch.randn(batch_size, C, 1, W)
# print("输入 x 的形状:", x.shape)  # 预期形状: (batch_size, 2*hidden_dim, 1, W)
# print("输入 y 的形状:", y.shape)  # 预期形状: (batch_size, 2*hidden_dim, H, 1)

# 初始化 DirectionalBiGRU 模型
model = DirectionalBiGRU(input_dim=C, hidden_dim=hidden_dim)

# 前向传播
output_x, output_y = model(x, y)


# print("---------------------------------------------------------------------------------")

# print("输出 x 的形状:", output_x.shape)  # 预期形状: (batch_size, 2*hidden_dim, 1, W)
# print("输出 y 的形状:", output_y.shape)  # 预期形状: (batch_size, 2*hidden_dim, H, 1)
