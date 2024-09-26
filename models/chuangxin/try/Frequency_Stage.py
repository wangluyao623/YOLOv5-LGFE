# import torch
# import torch.nn as nn

# class Frequedimy_Stage(nn.Module):
#     def __init__(self, dim): #dim = 64
#         super(Frequedimy_Stage, self).__init__()
#         self.process_pha = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, 1, 0),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(dim, dim, 1, 1, 0))
        
#         self.process_amp = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, 1, 0),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(dim, dim, 1, 1, 0))
        
#         self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        
#         self.conv_pha = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, 1, 0),
#             nn.LeakyReLU(0.1,inplace=True))

#         self.conv_amp = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, 1, 0),
#             nn.LeakyReLU(0.1,inplace=True))
        
#         # self.Neural = NRN()
#         self.Neural = nn.Conv2d(dim, 3, 1, 1, 0)
        
#         self.process_amp_NRN = nn.Sequential(
#             nn.Conv2d(3, dim, 1, 1, 0),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(dim, dim, 1, 1, 0))
        
#         self.process_pha_NRN = nn.Sequential(
#             nn.Conv2d(3, dim, 1, 1, 0),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(dim, dim, 1, 1, 0))
        
#         self.convoutfinal1 = nn.Conv2d(dim*2, dim, 1, 1, 0)
#         self.convoutfinal2 = nn.Conv2d(dim, dim, 1, 1, 0)

#     def forward(self, x):
#         xori = x ## [1,64,512,512]
#         _, _, H, W = x.shape
        
#         x_freq = torch.fft.rfft2(x, norm='backward') ##计算输入张量x的二维离散傅里叶变换 [1,64,512,257]
#         x_pha = torch.angle(x_freq)       ## [1,64,512,257]
#         x_amp = torch.abs(x_freq)         ## [1,64,512,257]
#         x_pha = self.process_pha(x_pha)   ## [1,64,512,257]
#         x_amp = self.process_amp(x_amp)   ## [1,64,512,257]
#         pha_NRN = self.Neural(x_pha)      ## [1,3,512,257]
#         pha_NRN = self.process_pha_NRN(pha_NRN) ## [1,64,512,257]
#         amp_NRN = self.Neural(x_amp)      ## [1,3,512,257]
#         amp_NRN = self.process_amp_NRN(amp_NRN) ## [1,64,512,257]
#         x_real = amp_NRN * torch.cos(pha_NRN)   ## [1,64,512,257]
#         x_imag = amp_NRN * torch.sin(pha_NRN)   ## [1,64,512,257]
#         x_spatial = torch.complex(x_real, x_imag) ## [1,64,512,257]
#         x_spatial = torch.fft.irfft2(x_spatial, s=(H, W), norm='backward') ## [1,64,512,512]
        
#         y_GAP = self.GAP(x_freq) ## [1,64,1,1]
#         y_GAP_pha = torch.angle(y_GAP)  ## [1,64,1,1]
#         y_GAP_amp = torch.abs(y_GAP)    ## [1,64,1,1]
#         y_GAP = y_GAP.real ## [1,64,1,1]
#         y_conv_pha = self.conv_pha(y_GAP) ## [1,64,1,1]
#         y_conv_amp = self.conv_amp(y_GAP) ## [1,64,1,1]
#         y_pha = y_GAP_pha * y_conv_pha    ## [1,64,1,1]
#         y_amp = y_GAP_amp * y_conv_amp    ## [1,64,1,1]
#         y_real = y_amp * torch.cos(y_pha) ## [1,64,1,1]
#         y_imag = y_amp * torch.sin(y_pha) ## [1,64,1,1]
#         y_channel = torch.complex(y_real, y_imag) ## [1,64,1,1]
#         y_channel = y_channel.expand_as(x_freq)   ## [1,64,512,257]
#         y_channel = torch.fft.irfft2(y_channel, s=(H, W), norm='backward') ## [1,64,512,512]
        
#         spa_cha = torch.cat([x_spatial, y_channel], dim=1) ## [1,128,512,512]
#         spa_cha = self.convoutfinal1(spa_cha) ## [1,64,512,512]
#         spa_cha = spa_cha + xori ## [1,64,512,512]
#         print("spa_cha:",spa_cha.shape)
#         out = self.convoutfinal2(spa_cha) ## [1,3,512,512]
#         return out
    
    
# # 设置随机种子以确保可复现性
# torch.manual_seed(0)

# # 实例化 DLK 模块
# dim = 32  # 输入通道数
# dlk_module = Frequedimy_Stage(dim)

# # 创建随机输入张量，形状为 (1, 32, 64, 64)
# x1 = torch.rand(1, dim, 64, 64)

# # 通过 DLK 模块进行前向传播
# output = dlk_module(x1)

# # 打印输出形状
# print("Output shape:", output.shape)












# #---------------------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.fft

# class FrequencyBranch(nn.Module):
#     def __init__(self, img_size, low_freq_radius=10):
#         """
#         :param img_size: 图像的尺寸 (height, width)
#         :param low_freq_radius: 控制低频掩膜的半径大小，用于提取低频信息
#         """
#         super(FrequencyBranch, self).__init__()
#         self.img_size = img_size
#         self.low_freq_radius = low_freq_radius
        
#         # 构建低频和高频的掩膜
#         self.low_freq_mask, self.high_freq_mask = self._create_frequency_masks()

#     def _create_frequency_masks(self):
#         """
#         构建频域中的高频和低频掩膜
#         :return: 低频掩膜和高频掩膜
#         """
#         h, w = self.img_size
#         low_freq_mask = torch.zeros(h, w)
        
#         # 计算频谱的中心位置
#         center_x, center_y = h // 2, w // 2
        
#         # 为低频掩膜中心区域赋值1
#         low_freq_mask[center_x-self.low_freq_radius:center_x+self.low_freq_radius, 
#                       center_y-self.low_freq_radius:center_y+self.low_freq_radius] = 1

#         # 高频掩膜为低频掩膜的补集
#         high_freq_mask = 1 - low_freq_mask
        
#         return low_freq_mask, high_freq_mask

#     def forward(self, x):
#         """
#         :param x: 输入图像，形状为 [batch_size, channels, height, width]
#         :return: 低频图像和高频图像
#         """
#         # 对输入图像进行傅里叶变换，得到频域表示
#         fft_image = torch.fft.fft2(x)

#         # 对频谱应用低频掩膜和高频掩膜，得到低频和高频成分
#         low_freq = fft_image * self.low_freq_mask.to(fft_image)
#         high_freq = fft_image * self.high_freq_mask.to(fft_image)

#         # 对低频和高频成分进行逆傅里叶变换，恢复到空间域
#         low_freq_image = torch.fft.ifft2(low_freq).real
#         high_freq_image = torch.fft.ifft2(high_freq).real

#         return low_freq_image, high_freq_image


# # 使用FrequencyBranch类
# if __name__ == "__main__":
#     # 创建随机输入图像，形状为 [batch_size, channels, height, width]
#     input_image = torch.randn(1, 32, 64, 64)  # 假设输入单通道图像，尺寸为64x64
    
#     # 实例化频率解耦类，假设低频半径为10
#     model = FrequencyBranch(img_size=(64, 64), low_freq_radius=10)

#     # 前向传播，得到低频图像和高频图像
#     low_freq_image, high_freq_image = model(input_image)

#     # 打印输出图像的形状
#     print("低频图像形状：", low_freq_image.shape)
#     print("高频图像形状：", high_freq_image.shape)


import torch
import torch.nn as nn

class FrequencyBranch(nn.Module):
    def __init__(self, low_freq_radius=10):
        """
        :param low_freq_radius: 控制低频掩膜的半径大小，用于提取低频信息
        """
        super(FrequencyBranch, self).__init__()
        self.low_freq_radius = low_freq_radius

    def _create_frequency_masks(self, img_size):
        """
        构建频域中的高频和低频掩膜
        :param img_size: 输入图像的尺寸 (height, width)
        :return: 低频掩膜和高频掩膜
        """
        h, w = img_size
        low_freq_mask = torch.zeros(h, w)
        
        # 计算频谱的中心位置
        center_x, center_y = h // 2, w // 2
        
        # 为低频掩膜中心区域赋值1
        low_freq_mask[center_x-self.low_freq_radius:center_x+self.low_freq_radius, 
                      center_y-self.low_freq_radius:center_y+self.low_freq_radius] = 1

        # 高频掩膜为低频掩膜的补集
        high_freq_mask = 1 - low_freq_mask
        
        return low_freq_mask, high_freq_mask

    def forward(self, x):
        """
        :param x: 输入图像，形状为 [batch_size, channels, height, width]
        :return: 低频图像和高频图像
        """
        # 自动获取输入图像的大小
        batch_size, channels, h, w = x.shape
        img_size = (h, w)
        
        # 构建低频和高频的掩膜
        low_freq_mask, high_freq_mask = self._create_frequency_masks(img_size)

        # 对输入图像进行傅里叶变换，得到频域表示
        fft_image = torch.fft.fft2(x)

        # 对频谱应用低频掩膜和高频掩膜，得到低频和高频成分
        low_freq = fft_image * low_freq_mask.to(fft_image)
        high_freq = fft_image * high_freq_mask.to(fft_image)

        # 对低频和高频成分进行逆傅里叶变换，恢复到空间域
        low_freq_image = torch.fft.ifft2(low_freq).real
        high_freq_image = torch.fft.ifft2(high_freq).real

        return low_freq_image, high_freq_image

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = FrequencyBranch(low_freq_radius=10)

    # 输入测试图像，假设是 1 个 3 通道的 128x128 图像
    x = torch.randn(1, 32, 64, 64)

    # 前向传播
    low_freq_image, high_freq_image = model(x)

    print("Low frequency image shape:", low_freq_image.shape)
    print("High frequency image shape:", high_freq_image.shape)













