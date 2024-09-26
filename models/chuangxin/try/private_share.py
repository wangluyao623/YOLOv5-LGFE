# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Private_Share_Module(nn.Module):
#     def __init__(self,
#                  coupled_number=32,
#                  n_feats=64,
#                  kernel_size=3):
#         super(Private_Share_Module, self).__init__()
#         self.n_feats = n_feats
#         self.coupled_number = coupled_number
#         self.kernel_size = kernel_size
#         self.kernel_shared_1=nn.Parameter(nn.init.kaiming_uniform(torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_depth_1=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_rgb_1=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_shared_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_depth_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_rgb_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        
#         self.bias_shared_1=nn.Parameter((torch.zeros(size=[self.coupled_number])))
#         self.bias_depth_1=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
#         self.bias_rgb_1=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        
#         self.bias_shared_2=nn.Parameter((torch.zeros(size=[self.coupled_number])))
#         self.bias_depth_2=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
#         self.bias_rgb_2=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        
#     def forward(self, feat_dlr, feat_rgb):
#         shortCut = feat_dlr
#         feat_dlr = F.conv2d(feat_dlr,
#                                   torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0), 
#                                   torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
#                                   padding=1)
#         feat_dlr = F.relu(feat_dlr, inplace=True)
#         feat_dlr = F.conv2d(feat_dlr,
#                                   torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0), 
#                                   torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
#                                   padding=1)
#         feat_dlr = F.relu(feat_dlr + shortCut, inplace=True)
#         shortCut = feat_rgb
#         feat_rgb = F.conv2d(feat_rgb,
#                                   torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0), 
#                                   torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
#                                   padding=1)
#         feat_rgb = F.relu(feat_rgb, inplace=True)
#         feat_rgb = F.conv2d(feat_rgb,
#                                   torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0), 
#                                   torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
#                                   padding=1)
#         feat_rgb = F.relu(feat_rgb + shortCut, inplace=True)
#         return feat_dlr, feat_rgb
    
# class Coupled_Encoder(nn.Module):
#     def __init__(self,
#                  n_feat=64,
#                  n_layer=4):
#         super(Coupled_Encoder, self).__init__()
#         self.n_layer = n_layer
#         self.init_deep=nn.Sequential( 
#                 nn.Conv2d(1, n_feat, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
#                 nn.ReLU(True),                               
#                 )  
#         self.init_rgb=nn.Sequential( 
#                 nn.Conv2d(3, n_feat, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
#                 nn.ReLU(True),                               
#                 )             
#         self.coupled_feat_extractor = nn.ModuleList([Private_Share_Module() for i in range(self.n_layer)])   

#     def forward(self, feat_dlr, feat_rgb):
#         feat_dlr = self.init_deep(feat_dlr)
#         feat_rgb = self.init_rgb(feat_rgb)
#         for layer in self.coupled_feat_extractor:
#             feat_dlr, feat_rgb = layer(feat_dlr, feat_rgb)
#         return feat_dlr, feat_rgb
    


# def test_coupled_encoder():
#     # 定义网络结构，n_layer 设置为 1
#     n_feat = 64  # 特征维度
#     n_layer = 1  # 网络层数设为1
#     coupled_encoder = Coupled_Encoder(n_feat=n_feat, n_layer=n_layer)
    
#     # 定义输入
#     batch_size = 1  # 批大小
#     height, width = 64, 64  # 输入图像的高和宽
    
#     # 随机生成输入：深度图 feat_dlr 和 RGB 图像 feat_rgb
#     feat_dlr = torch.randn(batch_size, 1, height, width)  # 深度图输入 (batch_size, 1, H, W)
#     feat_rgb = torch.randn(batch_size, 3, height, width)  # RGB 图像输入 (batch_size, 3, H, W)
    
#     # 前向传播
#     output_dlr, output_rgb = coupled_encoder(feat_dlr, feat_rgb)
    
#     # 打印输出的形状
#     print("Output shape for depth input:", output_dlr.shape)
#     print("Output shape for RGB input:", output_rgb.shape)

# # 执行测试
# test_coupled_encoder()






####################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Private_Share_Module(nn.Module):
#     def __init__(self,dim,kernel_size=3):
#         super(Private_Share_Module, self).__init__()
#         self.dim = dim
#         self.n_feats = dim
#         self.coupled_number = dim // 2
#         self.kernel_size = kernel_size
#         self.kernel_shared_1=nn.Parameter(nn.init.kaiming_uniform(torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_depth_1=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_rgb_1=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_shared_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_depth_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
#         self.kernel_rgb_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        
#         self.bias_shared_1=nn.Parameter((torch.zeros(size=[self.coupled_number])))
#         self.bias_depth_1=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
#         self.bias_rgb_1=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        
#         self.bias_shared_2=nn.Parameter((torch.zeros(size=[self.coupled_number])))
#         self.bias_depth_2=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
#         self.bias_rgb_2=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        
#     def forward(self, feat_dlr, feat_rgb):
#         shortCut = feat_dlr
#         feat_dlr = F.conv2d(feat_dlr,
#                                   torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0), 
#                                   torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
#                                   padding=1)
#         feat_dlr = F.relu(feat_dlr, inplace=True)
#         feat_dlr = F.conv2d(feat_dlr,
#                                   torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0), 
#                                   torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
#                                   padding=1)
#         feat_dlr = F.relu(feat_dlr + shortCut, inplace=True)
#         shortCut = feat_rgb
#         feat_rgb = F.conv2d(feat_rgb,
#                                   torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0), 
#                                   torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
#                                   padding=1)
#         feat_rgb = F.relu(feat_rgb, inplace=True)
#         feat_rgb = F.conv2d(feat_rgb,
#                                   torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0), 
#                                   torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
#                                   padding=1)
#         feat_rgb = F.relu(feat_rgb + shortCut, inplace=True)
#         return feat_dlr, feat_rgb
    
# class Coupled_Encoder(nn.Module):
#     def __init__(self,dim):
#         super(Coupled_Encoder, self).__init__()
#         self.dim = dim
#         self.n_feat = dim
#         self.init_deep=nn.Sequential( 
#                 nn.Conv2d(1, dim, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
#                 nn.ReLU(True),                               
#                 )  
#         self.init_rgb=nn.Sequential( 
#                 nn.Conv2d(3, dim, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
#                 nn.ReLU(True),                               
#                 )             
#         self.coupled_feat_extractor = Private_Share_Module()

#     def forward(self, feat_dlr, feat_rgb):
#         feat_dlr = self.init_deep(feat_dlr)
#         feat_rgb = self.init_rgb(feat_rgb)
#         for layer in self.coupled_feat_extractor:
#             feat_dlr, feat_rgb = layer(feat_dlr, feat_rgb)
#         return feat_dlr, feat_rgb
    


# def test_coupled_encoder():
#     # 定义网络结构，n_layer 设置为 1
#     n_feat = 64  # 特征维度
#     n_layer = 1  # 网络层数设为1
#     coupled_encoder = Coupled_Encoder(n_feat=n_feat, n_layer=n_layer)
    
#     # 定义输入
#     batch_size = 1  # 批大小
#     height, width = 64, 64  # 输入图像的高和宽
    
#     # 随机生成输入：深度图 feat_dlr 和 RGB 图像 feat_rgb
#     feat_dlr = torch.randn(batch_size, 1, height, width)  # 深度图输入 (batch_size, 1, H, W)
#     feat_rgb = torch.randn(batch_size, 3, height, width)  # RGB 图像输入 (batch_size, 3, H, W)
    
#     # 前向传播
#     output_dlr, output_rgb = coupled_encoder(feat_dlr, feat_rgb)
    
#     # 打印输出的形状
#     print("Output shape for depth input:", output_dlr.shape)
#     print("Output shape for RGB input:", output_rgb.shape)

# # 执行测试
# test_coupled_encoder()






import torch
import torch.nn as nn
import torch.nn.functional as F

class Private_Share_Module(nn.Module):
    def __init__(self,dim,kernel_size=3):
        super(Private_Share_Module, self).__init__()
        self.dim = dim
        self.n_feats = dim
        self.coupled_number = dim // 2
        self.kernel_size = kernel_size
        self.kernel_shared_1=nn.Parameter(nn.init.kaiming_uniform(torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_1=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_1=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_shared_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_2=nn.Parameter(nn.init.kaiming_uniform(torch.randn(size=[self.n_feats-self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        
        self.bias_shared_1=nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_1=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        self.bias_rgb_1=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        
        self.bias_shared_2=nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_2=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        self.bias_rgb_2=nn.Parameter((torch.zeros(size=[self.n_feats-self.coupled_number])))
        self.init_deep=nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
            nn.ReLU(True),                               
            )  
        self.init_rgb=nn.Sequential( 
            nn.Conv2d(dim, dim, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size
            nn.ReLU(True),                               
            )  
        
    def forward(self, feat_dlr, feat_rgb):
        feat_dlr = self.init_deep(feat_dlr)
        feat_rgb = self.init_rgb(feat_rgb)
        shortCut = feat_dlr
        feat_dlr = F.conv2d(feat_dlr,
                                  torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0), 
                                  torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
                                  padding=1)
        feat_dlr = F.relu(feat_dlr, inplace=True)
        feat_dlr = F.conv2d(feat_dlr,
                                  torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0), 
                                  torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
                                  padding=1)
        feat_dlr = F.relu(feat_dlr + shortCut, inplace=True)
        shortCut = feat_rgb
        feat_rgb = F.conv2d(feat_rgb,
                                  torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0), 
                                  torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
                                  padding=1)
        feat_rgb = F.relu(feat_rgb, inplace=True)
        feat_rgb = F.conv2d(feat_rgb,
                                  torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0), 
                                  torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
                                  padding=1)
        feat_rgb = F.relu(feat_rgb + shortCut, inplace=True)
        return feat_dlr, feat_rgb




def test_private_share_module():
    # 定义输入维度和卷积核大小
    dim = 32  # 输入的通道数
    kernel_size = 3  # 卷积核大小
    
    # 创建一个 Private_Share_Module 实例
    model = Private_Share_Module(dim=dim, kernel_size=kernel_size)
    
    # 定义输入
    batch_size = 1  # 批大小
    height, width = 64, 64  # 输入特征图的高和宽
    
    # 随机生成输入：深度图 feat_dlr 和 RGB 图像 feat_rgb
    feat_dlr = torch.randn(batch_size, dim, height, width)  # 深度图输入 (batch_size, dim, H, W)
    feat_rgb = torch.randn(batch_size, dim, height, width)  # RGB 图像输入 (batch_size, dim, H, W)
    
    # 前向传播
    output_dlr, output_rgb = model(feat_dlr, feat_rgb)
    
    # 打印输出的形状
    print("Output shape for depth input:", output_dlr.shape)
    print("Output shape for RGB input:", output_rgb.shape)

# 执行测试
test_private_share_module()










































