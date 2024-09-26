import torch
import torch.nn as nn


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        #print("inter_planes:",inter_planes)      # 4
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
            BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
            BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            # BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
            BasicConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, out_planes, kernel_size=1, stride=stride)
        )

        self.ConvLinear = BasicConv(4 * in_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):     #x: torch.Size([1, 32, 64, 64])
        # x0 = self.branch0(x)  # x0: torch.Size([1, 32, 64, 64])
        # x1 = self.branch1(x) # x1: torch.Size([1, 32, 64, 64])
        # x2 = self.branch2(x) # x2: torch.Size([1, 32, 64, 64])

        # out = torch.cat((x0, x1, x2), 1) # out1: torch.Size([1, 96, 64, 64])
        # out = self.ConvLinear(out) #out2: torch.Size([1, 32, 64, 64])
        # short = self.shortcut(x) #short: torch.Size([1, 32, 64, 64])
        # out = out * self.scale + short #out3: torch.Size([1, 32, 64, 64])
        # out = self.relu(out) #out4: torch.Size([1, 32, 64, 64])

        x0 = self.branch0(x)
        x1 = self.branch1(x0 + x)
        x2 = self.branch2(x0 + x1 + x)
        x3 = self.branch3(x0 + x1 + x2 + x)
        out = torch.cat((x0, x1, x2,x3), 1)
        out = self.ConvLinear(out)
        out = self.relu(out)

        return out
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    

# 总体的框架
class FourierTransformModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierTransformModule, self).__init__()

        # 频域分支
        self.freq_branch = FrequencyBranch(in_channels)

        # 空间域分支
        self.spatial_branch = SpatialBranch(in_channels)

        # 特征融合
        # self.fusion_layer = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.cv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.cv2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)



    def forward(self, x1, x2):
        freq_high, freq_low = self.freq_branch(x1) #1 32 64 64
        spatial_high, spatial_low = self.spatial_branch(x2) #1 32 64 64        1 32 62 62
        hl1 = torch.cat([freq_high, spatial_low], dim=1)
        hl2 = torch.cat([freq_low, spatial_high], dim=1)

        hl1 = self.cv1(hl1)
        hl2 = self.cv2(hl2)

        # 融合后经过卷积
        # output = self.fusion_layer(torch.cat([high_fused, low_fused], dim=1))
        output = hl1 + hl2

        return output

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

# 定义空间域分支
class SpatialBranch(nn.Module):
    def __init__(self, in_channels):
        super(SpatialBranch, self).__init__()

        # 高频处理层
        self.high_freq_conv = FEM(in_channels,in_channels)

        # 低频处理层
        self.low_freq_conv = BasicConv(in_channels,in_channels,3,1,1)

    def forward(self, x):
        # 空出空间域卷积部分
        high_freq = self.high_freq_conv(x)  # 通过卷积提取
        low_freq = self.low_freq_conv(x)   # 通过卷积提取

        return high_freq, low_freq


# 测试模块
if __name__ == "__main__":
    # 创建傅里叶变换模块
    model = FourierTransformModule(in_channels=32, out_channels=32)

    # 输入两个假设的图像特征 (batch_size, channels, height, width)
    x1 = torch.randn(1, 32, 64, 64)
    x2 = torch.randn(1, 32, 64, 64)

    # 前向传播
    output = model(x1, x2)

    print(output.shape)  # 输出特征的尺寸
