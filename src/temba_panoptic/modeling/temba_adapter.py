
from __future__ import annotations

import warnings

from typing import Iterable, List, Sequence

import torch
from torch import nn

class TemporalLocalModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, C]
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2)
        y = self.proj(y)
        return y

class DilatedTemporalMixer(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, C]
        y = self.norm(x)
        u, g = self.in_proj(y).chunk(2, dim=-1)
        u = u.transpose(1, 2)
        u = self.dwconv(u)
        u = u.transpose(1, 2)
        y = u * torch.sigmoid(g)
        y = self.out_proj(y)
        return y

class TembaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        local_kernel_size: int,
        local_dilation: int,
        dts_dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.local = TemporalLocalModule(dim, local_kernel_size, dilation=local_dilation)
        self.dts = DilatedTemporalMixer(dim, local_kernel_size, dilation=dts_dilation)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(x)
        x = x + self.dts(x)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class SingleScaleTembaAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        adapter_dim: int,
        adapter_depth: int,
        local_kernel_size: int,
        local_dilation: int,
        dts_dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, adapter_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(adapter_dim, in_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TembaBlock(
                    dim=adapter_dim,
                    local_kernel_size=local_kernel_size,
                    local_dilation=local_dilation,
                    dts_dilation=dts_dilation,
                    dropout=dropout,
                )
                for _ in range(adapter_depth)
            ]
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        if num_frames <= 1:
            warnings.warn(
                "TEMBA adapter received num_frames <= 1; returning the input unchanged.",
                RuntimeWarning,
            )
            return x

        bt, _, h, w = x.shape
        if bt % num_frames != 0:
            raise ValueError(
                f"Invalid TEMBA input: batch-time dimension {bt} is not divisible "
                f"by num_frames={num_frames}."
            )

        batch = bt // num_frames
        residual = x
        x = self.in_proj(x)
        c = x.shape[1]
        x = x.view(batch, num_frames, c, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(batch * h * w, num_frames, c)

        for block in self.blocks:
            x = block(x)

        x = x.view(batch, h, w, num_frames, c)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(bt, c, h, w)
        x = self.out_proj(x)
        return residual + x

class MultiScaleTembaAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        adapter_dim: int,
        adapter_depth: int,
        local_kernel_size: int,
        local_dilations: Sequence[int],
        dts_dilations: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        num_scales = min(len(local_dilations), len(dts_dilations))
        self.adapters = nn.ModuleList(
            [
                SingleScaleTembaAdapter(
                    in_channels=in_channels,
                    adapter_dim=adapter_dim,
                    adapter_depth=adapter_depth,
                    local_kernel_size=local_kernel_size,
                    local_dilation=int(local_dilations[i]),
                    dts_dilation=int(dts_dilations[i]),
                    dropout=dropout,
                )
                for i in range(num_scales)
            ]
        )

    def forward(self, features: Iterable[torch.Tensor], num_frames: int) -> List[torch.Tensor]:
        outputs = []
        for i, feat in enumerate(features):
            if i < len(self.adapters):
                outputs.append(self.adapters[i](feat, num_frames=num_frames))
            else:
                outputs.append(feat)
        return outputs
