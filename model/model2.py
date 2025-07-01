import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import math
from einops import repeat, rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

#------------------DCAattention----------
class DCAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个head的维度

        # QKV线性变换
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Q和K卷积路径（都使用k=3）
        self.conv_q3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv_k3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        # 拼接后压缩
        self.q_proj = nn.Linear(d_model * 2, d_model)
        self.k_proj = nn.Linear(d_model * 2, d_model)

        # 动态参数生成器
        self.tau_net = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(2 * d_model, n_heads, kernel_size=1)
        )
        self.delta_net = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(2 * d_model, n_heads, kernel_size=1)
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        # === Q处理 ===
        raw_Q = self.Wq(x)  # (B, L, d_model)
        q_input = raw_Q.permute(0, 2, 1)  # (B, d_model, L)
        Q3 = self.conv_q3(q_input)  # (B, d_model, L)
        Q_cat = torch.cat([q_input, Q3], dim=1).permute(0, 2, 1)  # (B, L, d_model*2)
        Q = self.q_proj(Q_cat).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)

        # === K处理 ===
        raw_K = self.Wk(x)  # (B, L, d_model)
        k_input = raw_K.permute(0, 2, 1)  # (B, d_model, L)
        K3 = self.conv_k3(k_input)  # (B, d_model, L)
        K_cat = torch.cat([k_input, K3], dim=1).permute(0, 2, 1)  # (B, L, d_model*2)
        K = self.k_proj(K_cat).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)

        # === V处理 ===
        V = self.Wv(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)

        # === tau & delta ===
        x_conv = x.permute(0, 2, 1)
        tau = torch.sigmoid(self.tau_net(x_conv)).transpose(1, 2).transpose(1, 2)  # (B, n_heads, L)
        delta = torch.sigmoid(self.delta_net(x_conv)).transpose(1, 2).transpose(1, 2)  # (B, n_heads, L)

        # === 注意力计算 ===
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)  # (B, n_heads, L, L)
        scores = scores * tau.unsqueeze(-1) + delta.unsqueeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)

#-------------原始attention----------------------
# class Attention(nn.Module):
#     def __init__(self, d_model, n_heads, dropout=0.1):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
#         self.n_heads = n_heads
#         self.d_k = d_model // n_heads  # 每个head的维度
#
#         # QKV线性变换
#         self.Wq = nn.Linear(d_model, d_model)
#         self.Wk = nn.Linear(d_model, d_model)
#         self.Wv = nn.Linear(d_model, d_model)
#
#         # 输出线性变换
#         self.out_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, mask=None):
#         B, L, _ = x.shape
#
#         # === Q, K, V处理 ===
#         Q = self.Wq(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)
#         K = self.Wk(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)
#         V = self.Wv(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)
#
#         # === 计算注意力得分 ===
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, n_heads, L, L)
#
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)  # 应用mask
#
#         attn = F.softmax(scores, dim=-1)  # 对得分进行softmax归一化
#         attn = self.dropout(attn)  # 应用dropout
#
#         # === 加权求和 ===
#         out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, -1)  # (B, L, d_model)
#
#         # 输出线性变换
#         return self.out_proj(out)


# ------------------ MoRFFeedForward ------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


#------------------ MoRFTransformerLayer ------------------
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = DCAttention(d_model, n_heads, dropout)
        #self.attention = Attention(d_model, n_heads, dropout)  # 使用标准的注意力机制
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 增强注意力 + 残差 + LayerNorm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 前馈网络 + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))



# ------------------ ChannelAttention ------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.silu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.silu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


# ------------------ SpatialAttention ------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


# ------------------ Mamba ------------------
class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=32,
            d_conv=3,
            expand=3,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = 512
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path

        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, 1, bias=bias)

        self.dwconv = nn.Conv2d(self.d_inner * 2, self.d_inner * 2, kernel_size=(5, 5), stride=1, padding=(2, 2),
                                groups=self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.sa = SpatialAttention(7)
        self.ca = ChannelAttention(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, dim, height, width = hidden_states.shape
        seqlen = height * width

        conv_state = None

        xz = self.dwconv(self.in_proj(hidden_states))
        xz = rearrange(xz, "b d h w -> b d (h w)")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states

            x, z = xz.chunk(2, dim=1)
            x = self.ca(x) * x
            z = self.sa(z) * z
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]

                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        out = rearrange(out, "b (h w) d -> b d h w", h=height, w=width)

        return out


# ------------------ MoRFPredictionBranch2 ------------------
class MoRFPredictionBranch2(nn.Module):
    def __init__(self, input_dim=1280, n_layers=4, d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.transformer = nn.ModuleList([
            TransformerLayer(input_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # Mamba模型输入维度与输出维度保持一致
        self.mamba = Mamba(d_model=input_dim)
        # 添加线性层，将输出维度降到512
        self.output_proj = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512)  # 稳定输出分布
        )

    def forward(self, x, mask=None, extract_mamba_features=False):
        # Transformer处理后维度保持为1280
        for layer in self.transformer:
            x = layer(x, mask)  # [B, 300, 1280]

        # Mamba processing - 支持3D输入
        if x.dim() == 3:  # [B, L, D] -> [B, D, L, 1]
            x = rearrange(x, "b l d -> b d l 1")
        x = self.mamba(x)  # [B, 512, L, 1]
        x = rearrange(x, "b d l 1 -> b l d")

        # 输出维度降到512
        x = self.output_proj(x)

        return x
