import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 每个通道单独卷积
            bias=False
        )
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x



class DPMKC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 方向卷积：分别对 H / W 两个维度建模（方向感知）
        self.conv_h = DWConv(in_channels, in_channels, kernel_size=(3,1), padding=(1,0))
        self.conv_w = DWConv(in_channels, in_channels, kernel_size=(1,3), padding=(0,1))

        # 多尺度卷积路径
        self.scale1 = DWConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.scale2 = DWConv(in_channels, in_channels, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        x = x.permute(0, 3, 1, 2).contiguous()
        direction_x = self.conv_w(self.conv_h(x))
        scale_x = self.scale2(self.scale1(x))
        # 特征融合（方向感知 + 多尺度）
        x_cat = torch.cat([direction_x, scale_x], dim=1)
        out = self.fusion(x_cat).permute(0, 2, 3, 1).contiguous()
        return out + residual




