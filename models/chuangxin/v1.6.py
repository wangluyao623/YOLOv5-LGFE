import torch
import torch.nn.functional as F
from torch import nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        #self.act=nn.SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)
        
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
  




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)




    def forward(self, x):

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1,-1,h,w)
        a_w = a_w.expand(-1, -1, h, w)

        # out = identity * a_w * a_h

        return a_w , a_h

class CoDEM2(nn.Module):

    def __init__(self,channel_dim):
        super(CoDEM2, self).__init__()

        self.channel_dim=channel_dim

        #特征连接后
        self.Conv3 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=2*self.channel_dim,kernel_size=3,stride=1,padding=1)
        #特征加和后
        # self.AvgPool = nn.functional.adaptive_avg_pool2d()
        self.Conv1 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        #最后输出
        # self.Conv1_ =nn.Conv2d(in_channels=3*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        self.GN1 = nn.GroupNorm(num_groups=2, num_channels=2*self.channel_dim)
        self.GN2 = nn.GroupNorm(num_groups=2, num_channels=self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        #我的注意力机制
        # self.coAtt_1 = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=16)
        #通道,kongjian注意力机制
        # self.cam =ChannelAttention(in_channels=self.channel_dim,ratio=16)
        # self.sam = SpatialAttention()


        #分两个分支new 参数量少了200w,不加逐点卷积参数量更少
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dim, out_channels=self.channel_dim, kernel_size=3, stride=1, padding=1, groups=self.channel_dim),
            # nn.Conv2d(in_channels=self.channel_dim, out_channels=self.channel_dim, kernel_size=1, stride=1, padding=0)
            )
        
        self.cv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_dim, out_channels=self.channel_dim, kernel_size=5, stride=1, padding=2, groups=self.channel_dim),
            # nn.Conv2d(in_channels=self.channel_dim, out_channels=self.channel_dim, kernel_size=1, stride=1, padding=0)
            ) 

        self.cv3 = nn.Conv2d(channel_dim, 2*channel_dim, kernel_size=3, stride=1, padding=1,groups=self.channel_dim)
        self.cv4 = nn.Conv2d(2*channel_dim,channel_dim, kernel_size=3, stride=1,padding=1,groups=self.channel_dim)
        self.cbam = CBAM(channel_dim,channel_dim)     

    def forward(self,x):
        x1 = self.cv1(x)
    
        x2 = self.cv2(x)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)

        x2_1 = self.cv3(x2)

        x2_2 = self.cv4(x2_1)

        x2_3 = x2_2 * x1

        x2_4 = self.cbam(x2_3)

        x2_5 = x2_2 + x2_4






        # B,C,H,W = x1.shape
        # f_d = torch.abs(x1-x2) #B,C,H,W
        f_c = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        z_c = self.ReLU(self.GN2(self.Conv1(self.ReLU(self.GN1(self.Conv3(f_c))))))

        # d_aw, d_ah = self.coAtt_1(f_d)
        # z_d = f_d * d_aw * d_ah


        out=z_c +x2_5

        return out










class DSCBottleNeck(nn.Module):
    def __init__(self, c1,c2,shortcut=True,g=1,e=0.5,ratio=16, kernel_size=7):
        super(DSCBottleNeck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1,g=g)
        self.add = shortcut and c1 == c2

        self.dem = CoDEM2(c2)


    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        out = self.dem(x2)
        out = F.interpolate(out, size=x1.size()[2:], mode='bilinear', align_corners=False)

        
        return x + out if self.add else out

class C3DSC(C3):  
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__(c1,c2,n,shortcut,g,e)
        self.m = nn.Sequential(*(DSCBottleNeck(c1, c1, shortcut, g, e=1.0) for _ in range(n)))
        #self.m=DSCBottleNeck(c1, c1, shortcut, g, e=1.0)

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        out = self.m(x)
        return out
