#随便拼的一个融合模块，外面是大核卷积，里面是一堆注意力（变化检测）里面的那个模块
#加在yolov5s上 在有第一个创新点的情况下还涨了一个点，家在Yolov5n上反而降了四个点


#---------------------------------------Fusion Block-------------------------------------------------------
class MyFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.cv1 = nn.Conv2d(in_channels=2*dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.fusion = CGAFusion(dim)

        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )


    def forward(self, x):   
        x12 = torch.cat([x[0], x[1]], dim=1)
        x = self.cv1(x12)


        f1 = self.att_conv1(x)#[1, 32, 64, 64]
        f2 = self.att_conv2(f1)#[1, 32, 64, 64]
        # print("att:",att.shape)
        result = self.fusion(f1,f2) #([1, 32, 64, 64])

        output = result + x
        
        return output


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8): #dim指的是输入tensor的通道数，该模块输入与输出相同
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, color_feat, graph_feat):
        initial = color_feat + graph_feat
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * color_feat + (1 - pattn2) * graph_feat
        result = self.conv(result)
        return result

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)

        cattn = self.ca(x_gap)
        return cattn



class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        # print(pattn1.shape)
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        # print(x2.shape)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

