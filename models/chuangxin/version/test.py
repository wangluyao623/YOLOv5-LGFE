import torch

# 假设输入特征图 x 的形状为 (batch_size, channels, height, width)
x = torch.randn(1, 32, 64, 64)  # 例如，batch_size=1, channels=64, height=64, width=64

# 使用切片操作将特征图在通道维度上切分为两个特征图
x1 = x[:, :x.size(1) // 2, :, :]  # 取前一半通道
x2 = x[:, x.size(1) // 2:, :, :]  # 取后一半通道

# 输出两个特征图的形状
print("x1 shape:", x1.shape)  # (batch_size, channels/2, height, width)
print("x2 shape:", x2.shape)  # (batch_size, channels/2, height, width)
