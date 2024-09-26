import torch
import torch.nn as nn
import torch.nn.functional as F

class FFT_Conv(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, n=10):
        super(FFT_Conv, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.n = n
        
        # 1x1卷积，用于通道的调整
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 通道归一化和激活
        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])
        self.sigmoid = nn.Sigmoid()
        
        # 全局平均池化（GAP）和全局最大池化（GMP）
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # 水平和垂直平均池化 (HAP, VAP)
        self.hap = nn.AdaptiveAvgPool2d((None, 1))  # 水平池化
        self.vap = nn.AdaptiveAvgPool2d((1, None))  # 垂直池化

        self.h_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5))  # 水平卷积
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0))  # 垂直卷积
        
        
        # 融合权重（WFU）
        self.w_fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x): #1 32 64 64
        # 傅里叶变换 -> 频域
        fft_image = torch.fft.fft2(x)#1 32 64 64
        fft_image = torch.fft.fftshift(fft_image)#1 32 64 64

        # 取傅里叶变换后的实部或模值来进行后续操作
        fft_image = torch.abs(fft_image)  # 或者用 .real 只取实部#1 32 64 64
        
        # 垂直和水平池化
        v_ap = self.vap(fft_image) #[1, 32, 1, 64]
        print("v_ap:",v_ap.shape)
        v_ap = self.v_conv(v_ap) #[1, 32, 2, 64]
        print("v_ap:",v_ap.shape)
        h_ap = self.hap(fft_image)#[1, 32, 64, 1] 
        print("h_ap:",h_ap.shape)
        h_ap = self.h_conv(h_ap) #[1, 32, 64, 2]
        print("h_ap:",h_ap.shape)

        # 全局平均池化和全局最大池化
        g_ap = self.gap(fft_image) #[1, 32, 1, 1]
        g_mp = self.gmp(fft_image) #[1, 32, 1, 1]

        # 将GAP和GMP融合为一个
        fusion = g_ap + g_mp  #([1, 32, 1, 1])
        # print("fusion:",fusion.shape)

        # 1x1卷积降维
        fusion = self.conv_1x1(fusion) #[1, 32, 1, 1]
        fusion = self.layer_norm(fusion)#[1, 32, 1, 1]
        fusion = self.sigmoid(fusion)#[1, 32, 1, 1]
        
        # 将垂直和水平池化结果与融合特征相乘
        v_ap_weighted = v_ap * fusion #[1, 32, 2, 64]
        h_ap_weighted = h_ap * fusion #[1, 32, 64, 2]
        
        # 融合频域信息
        combined = v_ap_weighted + h_ap_weighted
        combined = self.w_fusion(combined)
        
        # 逆傅里叶变换 -> 回到时域
        combined = torch.fft.ifftshift(combined)
        output_image = torch.fft.ifft2(combined).real

        return output_image
    



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

        # 引入 FFT_Conv 作为共享卷积部分
        self.shared_fft_conv = FFT_Conv(dim)


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





# 测试
if __name__ == "__main__":
    batch_size, channels, height, width = 1, 32, 64, 64
    x = torch.randn(batch_size, channels, height, width)  # 模拟输入

    model = FFT_Conv(in_channels=channels)
    output = model(x)
    print(output.shape)
