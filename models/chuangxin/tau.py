import torch
import torch.nn as nn
import torch.nn.functional as F

class TAU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TAU, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dw_d_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        # Statical Attention
        sa = self.dw_conv(x)
        sa = self.dw_d_conv(sa)
        sa = self.pointwise_conv(sa)
        
        # Dynamical Attention
        b, c, _, _ = x.size()
        da = self.avg_pool(x).view(b, c)
        da = self.fc(da).view(b, c, 1, 1)
        da = da.sigmoid()
        
        # Combining both attentions
        out = sa * da
        
        return out

# Example usage
input_tensor = torch.randn(8, 64, 32, 32)  # Batch size of 8, 64 channels, 32x32 image
tau_module = TAU(in_channels=64, out_channels=64)
output_tensor = tau_module(input_tensor)
print(output_tensor.shape)
