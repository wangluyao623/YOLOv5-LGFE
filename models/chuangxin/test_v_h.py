
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp


from torchvision import models










class RCA(nn.Module):
    def __init__(self, inp,  kernel_size=1, ratio=1, band_kernel_size=11,dw_size=(1,1), padding=(0,0), stride=1, square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size//2, groups=inp)
        

        gc=inp//ratio
        self.excite = nn.SequentialCell(
                nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc),
                nn.Sigmoid()
            )
    
    def sge(self, x):
        #[N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w #.repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather) # [N, 1, C, 1]
        
        return ge

    def forward(self, x):
        loc=self.dwconv_hw(x)
        att=self.sge(x)
        out = att*loc
        
        return out





class DLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid =  nn.Sigmoid()
        self.rca = RCA(inp=dim)
    
        # self.fc1 = nn.Linear(dim, dim)
        self.cv1 = nn.Conv2d(in_channels=2*dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.cv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x1,x2): #1,32,64,64||1,32,64,64  
        x1_1_a = self.avg_pool(x1) #1,32,1,1
        x1_1_m = self.max_pool(x1) #1,32,1,1

        x1_2 = torch.cat([x1_1_a, x1_1_m], dim=1) #1,64,1,1

        x1_3 = self.cv1(x1_2) #1,32,1,1



        x2_1_a = self.avg_pool(x2) #1,32,1,1
        x2_1_m = self.max_pool(x2) #1,32,1,1

        x2_2 = torch.cat([x2_1_a, x2_1_m], dim=1) #1,64,1,1

        x2_3 = self.cv1(x2_2) #1,32,1,1
        

        mul1 = x1_3 * x2_3 #1,32,1,1

        
        mul1 = self.cv2(mul1)
        mul1 = self.sigmoid(mul1)#1,32,1,1
        

        

        f1 = mul1*x1 #1,32,64,64
        f1 = self.cv2(f1)
        f2 = mul1*x2
        f2 = self.cv2(f1)
    #----------------------------------------------------------------------------


        att = torch.cat([f1, f2], dim=1) #1,64,64,64
        avg_att = torch.mean(att, dim=1, keepdim=True) #1,1,64,64
        max_att,_ = torch.max(att, dim=1, keepdim=True) #1,1,64,64

        att = torch.cat([avg_att, max_att], dim=1)#1,2,64,64
        att = self.spatial_se(att) #1,2,64,64
        

        out1 = f1 * att[:,0,:,:].unsqueeze(1) 
        out1 = out1 + f1
        out1 = self.cv2(out1)
        out2 = f2 * att[:,1,:,:].unsqueeze(1)
        out2 = out2 + f2
        out2 = self.cv2(out2)
        output = out1 + out2
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
output = dlk_module(x1,x2)

# 打印输出形状
print("Output shape:", output.shape)