
class TAU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TAU, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dw_d_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # Statical Attention                   #x [1*64*32*32]
        sa = self.dw_conv(x)                  #sa [1*64*32*32]
        sa = self.dw_d_conv(sa)               #sa [1*64*32*32]
        sa = self.pointwise_conv(sa)          #sa [1*64*32*32]
        
        # Dynamical Attention
        b, c, _, _ = x.size()
        da = self.avg_pool(x).view(b, c)      #da [1, 64]
        da = self.fc(da).view(b, c, 1, 1)     #da [1, 64,1,1]
       
        da = da.sigmoid()                     ##da [1, 64,1,1]


        # Combining both attentions
        out = sa * da #1，64，32，32
        
        return out

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class CB2d(nn.Module):
    def __init__(self, inplanes, pool='att', fusions=['channel_add', 'channel_mul']):
        super(CB2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = inplanes // 2
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':  # iscyy
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(3)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term #[1,32,64,64]
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context) #[1,32,1,1]
            out = out + channel_add_term
        return out
    
#-----------------------------------------------------------------------------------------------


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


class CoDEM2(nn.Module):

    def __init__(self,channel_dim):
        super(CoDEM2, self).__init__()

        self.channel_dim=channel_dim

        # #特征连接后
        # self.Conv3 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=2*self.channel_dim,kernel_size=3,stride=1,padding=1)
        # #特征加和后
        # # self.AvgPool = nn.functional.adaptive_avg_pool2d()
        # self.Conv1 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        #最后输出
        # self.Conv1_ =nn.Conv2d(in_channels=3*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        self.GN1 = nn.GroupNorm(num_groups=2, num_channels=2*self.channel_dim)
        self.GN2 = nn.GroupNorm(num_groups=2, num_channels=2*self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)

        #分两个分支new 参数量少了200w,不加逐点卷积参数量更少2
        self.cv1 = nn.Conv2d(in_channels=self.channel_dim, out_channels=self.channel_dim, kernel_size=3, stride=2, padding=1, groups=self.channel_dim)
        
        self.cv2 = nn.Conv2d(in_channels=self.channel_dim, out_channels=self.channel_dim, kernel_size=3, stride=2, padding=1, groups=self.channel_dim)
        
        self.cv3 = nn.Conv2d(in_channels=self.channel_dim, out_channels=self.channel_dim, kernel_size=5, stride=1, padding=2, groups=self.channel_dim)
           
        
        self.cv4 = nn.Conv2d(2*channel_dim, 2*channel_dim, kernel_size=3, stride=1, padding=1,groups=self.channel_dim)
        self.cv5 = nn.Conv2d(2*channel_dim,2*channel_dim, kernel_size=3, stride=1,padding=1,groups=self.channel_dim)
        self.cv6 = nn.Conv2d(channel_dim, 2*channel_dim,3, 1, 1, groups=self.channel_dim)
        self.cv7 = nn.Conv2d(2*channel_dim,channel_dim,kernel_size=1)
        self.tau = TAU(2*channel_dim,channel_dim)#1,64,1,1
        self.cb2d = CB2d(channel_dim)

  

    def forward(self,x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x)#1，32，64，64

        x2_1 = torch.abs(x1-x2) #B,C,H,W
        x2_2 = self.cv6(x2_1)#1*64*32*32
        x2_3 = self.tau(x2_2)#1,64,32,32
        x1_1 = torch.cat((x1, x2), dim=1)  # B,2C,H,W #1,64,32,32
        # xxx1 = self.ReLU(self.GN1(self.cv4(x1_1)))
        # xxx2 = self.cv5(xxx1)
        # xxx3 = self.GN2(xxx2)
        x1_2 = self.ReLU(self.GN2(self.cv5(self.ReLU(self.GN1(self.cv4(x1_1))))))#1，64，32，32

        x2_4 = x2_3+x1_2#1，64，32，32
        x2_5 = self.cv7(x2_4)#1，32，32，32

        x2_5 = F.interpolate(x2_5, scale_factor=2, mode='nearest')#1，32，64，64

        x3_1 = self.cb2d(x3)#1，32，64，64
        # print("x3_1",x3_1.size(),"x2_5:",x2_5.size())
        x3_2 = x3_1 + x2_5

        out = x3_2

        return out


import torch.nn.functional as F

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


