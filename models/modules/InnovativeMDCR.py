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

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class InnovativeMDCR(nn.Module):
    def __init__(self, in_features, out_features, rate=[1, 3, 5, 7]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        inter_channels = out_features // 4  # 中间通道数
        # 多尺度分支
        self.branches = nn.ModuleList([
            DWConv(
                in_channels=in_features // 4,
                out_channels=inter_channels,
                kernel_size=3,
                padding=rate[i],
                dilation=rate[i]
            ) for i in range(4)
        ])
        # 通道注意力
        self.attention = ChannelAttention(channels=out_features)
        # 残差连接的点态卷积
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_features, out_features, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # 残差连接（可选）
        if in_features != out_features:
            self.residual = DWConv(in_features, out_features, 3, 1)
            # self.residual = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            #                               nn.InstanceNorm2d(out_features), nn.LeakyReLU(inplace=True))
        else:
            self.residual = nn.Identity()

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 重塑并重排通道
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # 交换 groups 和 channels_per_group
        x = x.view(batch_size, num_channels, height, width)
        return x


    def forward(self, x):
        # 分割通道
        x_skip = x
        x = torch.chunk(x, 4, dim=1)

        # 多尺度特征提取
        x = [self.branches[i](x[i]) for i in range(4)]

        # 拼接与通道混洗
        x = torch.cat(x, dim=1)
        x = self.channel_shuffle(x, 4)
        # 通道注意力
        x = self.attention(x)
        # 输出卷积
        x = self.out_conv(x)
        res = self.residual(x_skip)
        x = x + res
        return x

class CPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DMSC_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 方向卷积：分别对 H / W 两个维度建模（方向感知）
        self.conv_h = DWConv(in_channels, in_channels, kernel_size=(3,1), padding=(1,0))
        self.conv_w = DWConv(in_channels, in_channels, kernel_size=(1,3), padding=(0,1))

        # 多尺度卷积路径（1x1 + 3x3 + 5x5）用于局部-全局感受野融合
        self.scale1 = DWConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.scale2 = DWConv(in_channels, in_channels, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        direction_x = self.conv_w(self.conv_h(x))
        scale_x = self.scale2(self.scale1(x))
        # 特征融合（方向感知 + 多尺度）
        x_cat = torch.cat([direction_x, scale_x], dim=1)
        out = self.fusion(x_cat)
        return out + residual

class DMSC_3(nn.Module):
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

class DMSC_4(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 方向卷积：分别对 H / W 两个维度建模（方向感知）
        self.conv_h = DWConv(in_channels//2, in_channels//2, kernel_size=(3,1), padding=(1,0))
        self.conv_w = DWConv(in_channels//2, in_channels//2, kernel_size=(1,3), padding=(0,1))

        # 多尺度卷积路径
        self.scale1 = DWConv(in_channels//2, in_channels//2, kernel_size=3, padding=1)
        self.scale2 = DWConv(in_channels//2, in_channels//2, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        x = x.permute(0, 3, 1, 2).contiguous()
        splits = torch.chunk(x, 2, dim=1)
        direction_x = self.conv_w(self.conv_h(splits[0]))
        scale_x = self.scale2(self.scale1(splits[1]))
        # 特征融合（方向感知 + 多尺度）
        x_cat = torch.cat([direction_x, scale_x], dim=1)
        out = self.fusion(x_cat).permute(0, 2, 3, 1).contiguous()
        return out + residual


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class InnovativeMDCR2(nn.Module):
    def __init__(self, in_features, out_features, rate=[1, 3, 5, 7]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        inter_channels = out_features // 4  # 中间通道数
        # 多尺度分支
        self.branches = nn.ModuleList([
            DWConv(
                in_channels=in_features // 4,
                out_channels=inter_channels,
                kernel_size=3,
                padding=rate[i],
                dilation=rate[i]
            ) for i in range(4)
        ])
        self.eca = ECA(out_features)
        self.sa = SpatialAttentionModule()
        self.norm = nn.InstanceNorm2d(out_features)
        self.act = nn.LeakyReLU(inplace=True)
        # 残差连接（可选）
        if in_features != out_features:
            # self.residual = DWConv(in_features, out_features, 1, 0)
            # self.residual = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)
            self.residual = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
                                          nn.InstanceNorm2d(out_features), nn.LeakyReLU(inplace=True))
        else:
            self.residual = nn.Identity()

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 重塑并重排通道
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # 交换 groups 和 channels_per_group
        x = x.view(batch_size, num_channels, height, width)
        return x


    def forward(self, x):
        # 分割通道
        x_skip = x
        x = torch.chunk(x, 4, dim=1)
        # 多尺度特征提取
        x = [self.branches[i](x[i]) for i in range(4)]
        # 拼接与通道混洗
        x = torch.cat(x, dim=1)
        x = self.channel_shuffle(x, 4)
        # 通道注意力
        x = self.eca(x)
        x = self.sa(x)
        x = self.norm(x)
        x = self.act(x)
        res = self.residual(x_skip)
        x = x + res
        return x


class InnovativeMDCR3(nn.Module):
    def __init__(self, in_features, out_features, rate=[1, 3, 5, 7]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        inter_channels = out_features // 4  # 中间通道数
        self.fusion_conv = nn.Conv2d(in_features * 2, in_features, kernel_size=1)
        # 多尺度分支
        self.branches = nn.ModuleList([
            DWConv(
                in_channels=in_features // 4,
                out_channels=inter_channels,
                kernel_size=3,
                padding=rate[i],
                dilation=rate[i]
            ) for i in range(4)
        ])
        self.eca = ECA(out_features)
        self.sa = SpatialAttentionModule()
        self.norm = nn.InstanceNorm2d(out_features)
        self.act = nn.LeakyReLU(inplace=True)
        # 残差连接（可选）
        if in_features != out_features:
            # self.residual = DWConv(in_features, out_features, 1, 0)
            self.residual = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)
            # self.residual = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            #                               nn.InstanceNorm2d(out_features), nn.LeakyReLU(inplace=True))
        else:
            self.residual = nn.Identity()

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 重塑并重排通道
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # 交换 groups 和 channels_per_group
        x = x.view(batch_size, num_channels, height, width)
        return x


    def forward(self, x):
        x = self.fusion_conv(x)
        # 分割通道
        x_skip = x
        x = torch.chunk(x, 4, dim=1)
        # 多尺度特征提取
        x = [self.branches[i](x[i]) for i in range(4)]
        # 拼接与通道混洗
        x = torch.cat(x, dim=1)
        x = self.channel_shuffle(x, 4)
        # 通道注意力
        x = self.eca(x)
        x = self.sa(x)
        x = self.norm(x)
        x = self.act(x)
        res = self.residual(x_skip)
        x = x + res
        return x


class InnovativeMDCR4(nn.Module):
    def __init__(self, in_features, out_features, rate=[1, 3, 5, 7]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        inter_channels = out_features // 4  # 中间通道数
        # 多尺度分支
        self.branches = nn.ModuleList([
            DWConv(
                in_channels=in_features // 4,
                out_channels=inter_channels,
                kernel_size=3,
                padding=rate[i],
                dilation=rate[i]
            ) for i in range(4)
        ])
        self.eca = ECA(out_features)
        self.sa = SpatialAttentionModule()
        self.norm = nn.InstanceNorm2d(out_features)
        self.act = nn.LeakyReLU(inplace=True)
        # 残差连接（可选）
        if in_features != out_features:
            # self.residual = DWConv(in_features, out_features, 1, 0)
            self.residual = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)
            # self.residual = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            #                               nn.InstanceNorm2d(out_features), nn.LeakyReLU(inplace=True))
        else:
            self.residual = nn.Identity()

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # 重塑并重排通道
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # 交换 groups 和 channels_per_group
        x = x.view(batch_size, num_channels, height, width)
        return x


    def forward(self, x):
        # 分割通道
        x_skip = x
        x = self.channel_shuffle(x, 4)
        x = torch.chunk(x, 4, dim=1)
        # 多尺度特征提取
        x = [self.branches[i](x[i]) for i in range(4)]
        # 拼接与通道混洗
        x = torch.cat(x, dim=1)
        # 通道注意力
        x = self.eca(x)
        x = self.sa(x)
        x = self.norm(x)
        x = self.act(x)
        res = self.residual(x_skip)
        x = x + res
        return x




