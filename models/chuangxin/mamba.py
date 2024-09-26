import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp


from torchvision import models



class GroupMambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=1, d_conv=3, expand=1, reduction=16):
        super().__init__()

        num_channels_reduced = input_dim // reduction
        self.fc1 = nn.Linear(input_dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba_g1 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g2 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g3 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g4 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, H, W):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, N, C = x.shape
        x = self.norm(x)

        # Channel Affinity
        z = x.permute(0, 2, 1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        x = rearrange(x, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)

        # Four scans applied to 4 different directions, each is applied for N/4 channels
        x_mamba1 = self.mamba_g1(x1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1)
        x_mamba2 = self.mamba_g2(x2, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2)
        x_mamba3 = self.mamba_g3(x3, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3)
        x_mamba4 = self.mamba_g4(x4, CrossScan=CrossScan_4, CrossMerge=CrossMerge_4)

        # Combine all feature maps
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1) * self.skip_scale * x

        x_mamba = rearrange(x_mamba, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)

        # Channel Modulation
        x_mamba = x_mamba * fc_out_2.unsqueeze(1)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)

        return x_mamba
    
