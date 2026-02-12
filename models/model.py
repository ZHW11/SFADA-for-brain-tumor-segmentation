# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.utils import look_up_option

from .modules.Innovative import DPMKC
from .modules.StructureSSM_mySP import StructureAwareSSM
from monai.networks.nets.swin_unetr import PatchMergingV2
from monai.networks.blocks import MLPBlock, PatchEmbed
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = StructureAwareSSM(
            d_model=dim,
            d_state=1,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        b, h, w, c = x.shape
        x_skip = x
        assert c == self.dim
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        out = x_mamba + x_skip
        return out


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            PatchEmbed(2, in_chans, dims[0], spatial_dims=2)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = PatchMergingV2(dims[i], spatial_dims=2)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.dpmkcs = nn.ModuleList()
        for i in range(4):
            dpmkc = DPMKC(dims[i])
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            self.dpmkcs.append(dpmkc)

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.LayerNorm(dims[i_layer])  # 改为2D归一化
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MLPBlock(dims[i_layer], 4 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i == 0:
                x = x.permute(0, 2, 3, 1).contiguous()
            x = self.dpmkcs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                x_out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(x_out)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class SFADA_Net(nn.Module):
    def __init__(
            self,
            in_chans=4,
            out_chans=4,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size=None,
            norm_name="instance",
            res_block=True,
            spatial_dims=2,  # 修改为2D
            device = torch.device("cuda:0"),
            deep_supervision=False,
    ):
        super().__init__()

        self.hidden_size = feat_size[3] * 2
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision

        self.vit = MambaEncoder(
            in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.n = 4
        self.bottle = nn.ModuleList()
        current_size = self.feat_size[0]
        for i in range(self.n):
            next_size = current_size * 2  # 自定义变化逻辑
            self.bottle.append(
                 UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=current_size,
                    out_channels=next_size,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=res_block,
                 )
            )
            current_size = next_size  # 更新当前通道数

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder0 = UnetrBasicBlock(2, self.feat_size[0], self.feat_size[0], kernel_size=3,
                                        stride=1, res_block=res_block, norm_name=norm_name)
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=self.out_chans
        )


    def forward(self, x_in):
        out_list = []
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        for i, layer in enumerate(self.bottle):
            outs[i] = layer(outs[i])
        x = self.decoder4(outs[3], outs[2])
        x = self.decoder3(x, outs[1])
        x = self.decoder2(x, outs[0])
        x = self.decoder1(x, enc1)
        x = self.decoder0(x)
        x = self.out(x)
        return x



if __name__ == "__main__":

    x = torch.randn(1, 128, 128, 32).cuda()
    block = StructureAwareSSM(32, 1).cuda()

    y = block(x)
    print(y.shape)


