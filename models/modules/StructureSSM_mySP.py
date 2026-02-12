import torch.nn as nn
import torch
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from .spatial_utils import selective_scan_state_flop_jit, selective_scan_fn, Stem, DownSampling


class StateFusion(nn.Module):
    def __init__(self, dim, use_pixel_weights=True):
        super().__init__()
        self.dim = dim
        self.group_dim = dim // 4  # 每组通道数
        self.use_pixel_weights = use_pixel_weights

        # 为每组定义不同的卷积核
        self.kernel_3_g1 = nn.Parameter(torch.ones(self.group_dim, 1, 3, 3))
        self.kernel_3_g2 = nn.Parameter(torch.ones(self.group_dim, 1, 3, 3))
        self.kernel_3_g3 = nn.Parameter(torch.ones(self.group_dim, 1, 3, 3))
        self.kernel_3_g4 = nn.Parameter(torch.ones(self.group_dim, 1, 3, 3))

        if use_pixel_weights:
            # 方法1：为每个像素点生成权重
            self.weight_conv = nn.Sequential(
                nn.Conv2d(dim, dim//8, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(dim//8, 4, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            # 方法2：固定权重参数
            self.alpha = nn.Parameter(torch.ones(4)/4)

    @staticmethod
    def padding(input_tensor, padding):
        return F.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):
        B, C, H, W = h.shape  # 获取输入形状
        # 分组处理
        h_groups = torch.chunk(h, 4, dim=1)  # 沿通道维度分成4组，每组形状 [B, C//4, H, W]
        h1 = F.conv2d(self.padding(h_groups[0], (1, 1, 1, 1)),
                      self.kernel_3_g1, padding=0, dilation=1, groups=self.group_dim)
        h2 = F.conv2d(self.padding(h_groups[1], (2, 2, 2, 2)),
                      self.kernel_3_g2, padding=0, dilation=2, groups=self.group_dim)
        h3 = F.conv2d(self.padding(h_groups[2], (3, 3, 3, 3)),
                      self.kernel_3_g3, padding=0, dilation=3, groups=self.group_dim)
        h4 = F.conv2d(self.padding(h_groups[3], (4, 4, 4, 4)),
                      self.kernel_3_g4, padding=0, dilation=4, groups=self.group_dim)
        # 将结果拼接回原始通道数 [B, C, H, W]
        if self.use_pixel_weights:
            # 方法1：像素级权重
            weights = self.weight_conv(h)  # [B, 4, H, W]
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)  # 手动归一化
            h1 = h1 * weights[:, 0:1, :, :]  # [B, C//4, H, W]
            h2 = h2 * weights[:, 1:2, :, :]
            h3 = h3 * weights[:, 2:3, :, :]
            h4 = h4 * weights[:, 3:4, :, :]
            out = torch.cat([h1, h2, h3, h4], dim=1)  # [B, C, H, W]
        else:
            # 方法2：固定权重
            alpha = F.softmax(self.alpha, dim=0)  # [4]
            # 加权合并并恢复通道数
            out = torch.cat([
                (h1 * alpha[0]).unsqueeze(1),
                (h2 * alpha[1]).unsqueeze(1),
                (h3 * alpha[2]).unsqueeze(1),
                (h4 * alpha[3]).unsqueeze(1)
            ], dim=1)  # [B, 4, C//4, H, W]
            out = out.reshape(B, C, H, W)  # [B, C, H, W]

        return out


class StructureAwareSSM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=1,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            use_pixel_weights=True,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                     **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn
        # self.prompt_alpha = nn.Parameter(torch.tensor(0.2))
        # self.inner_rank = 128
        # self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        # self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)
        self.num_tokens = 128
        # self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        # self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)
        self.embedding = nn.Embedding(self.num_tokens, self.d_state)
        self.embedding.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)
        self.route = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 3),
            nn.GELU(),
            nn.Linear(self.d_model // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )
        self.state_fusion = StateFusion(self.d_inner * self.d_state, use_pixel_weights=use_pixel_weights)

        # Semantic Prompting
        self.num_prompt_tokens = 8
        self.prompt_proj = nn.Linear(self.d_inner, self.d_state)
        self.prompt_router = nn.Conv2d(self.d_inner, self.num_prompt_tokens, kernel_size=1)
        self.token_bank_proj = nn.Linear(self.d_inner, self.d_state)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                bias=True, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init == "random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init == "simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init == "zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init == "random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor, prompt):
        B, C, H, W = x.shape
        L = H * W
        xs = x.view(B, -1, L)

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)
        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")
        y = h * (Cs + prompt)
        y = y + xs * Ds.view(-1, 1)
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # full_embedding = self.embeddingB.weight @ self.embeddingA.weight  # [128, C]
        full_embedding = self.embedding.weight  # [128, C]
        pred_route = self.route(x.view(B, H*W, C))  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]
        prompt = torch.matmul(cls_policy, full_embedding).view(B, H*W, self.d_state)
        prompt = prompt.permute(0, 2, 1)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d(x))
        y = self.ssm(x, prompt)
        y = rearrange(y, 'b d (h w)-> b h w d', h=H, w=W)
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y