##两个阶段融合，都是做空间

import torch
import torch.nn as nn
#-----------------------------------------Fusion------------------------------------------------------
class DLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
    

        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.sigmoid =  nn.Sigmoid()

    def forward(self, x1,x2):
        t1 = x1
        t2 = x2   
      
        avg_att_x1_1 = torch.mean(x1, dim=1, keepdim=True) #1,1,64,64
        max_att_x1_1,_ = torch.max(x1, dim=1, keepdim=True) #1,1,64,64
        avg_max_cat_x1_1 = torch.cat([avg_att_x1_1, max_att_x1_1], dim=1)#1,2,64,64
        avg_max_cat_x1_1 = self.cv1( avg_max_cat_x1_1)#1,1,64*64

        avg_att_x2_1 = torch.mean(x2, dim=1, keepdim=True) #1,1,64,64
        max_att_x2_1,_ = torch.max(x2, dim=1, keepdim=True) #1,1,64,64
        avg_max_cat_x2_1 = torch.cat([avg_att_x2_1, max_att_x2_1], dim=1)#1,2,64,64
        avg_max_cat_x2_1 = self.cv1( avg_max_cat_x2_1)#1,1,64*64

        output1 = avg_max_cat_x1_1 * avg_max_cat_x2_1 #1,1,64*64
        # print(output.shape)

        output1 = self.sigmoid(output1)

        avg_max_cat_x1_2 = output1 + avg_max_cat_x1_1#1,1,64,64
        avg_max_cat_x2_2 = output1 + avg_max_cat_x2_1#1,1,64,64

        att = torch.cat([ avg_max_cat_x1_2, avg_max_cat_x2_2], dim=1)#1*2*64*64



        avg_att = torch.mean(att, dim=1, keepdim=True) #1,1,64,64
        max_att,_ = torch.max(att, dim=1, keepdim=True) #1,1,64,64

        att = torch.cat([avg_att, max_att], dim=1)#1,2,64,64
        att = self.spatial_se(att) #1,2,64,64
     

        output = x1 * att[:,0,:,:].unsqueeze(1) + x2 * att[:,1,:,:].unsqueeze(1)
        return output










# 设置随机种子以确保可复现性
torch.manual_seed(0)

# 实例化 DLK 模块
dim = 32  # 输入通道数
dlk_module = DLK(dim)

# 创建随机输入张量，形状为 (1, 32, 64, 64)
x1 = torch.rand(1, dim, 64, 64)
x2 = torch.rand(1, dim, 64, 64)

# 通过 DLK 模块进行前向传播
output = dlk_module(x1, x2)

# 打印输出形状
print("Output shape:", output.shape)