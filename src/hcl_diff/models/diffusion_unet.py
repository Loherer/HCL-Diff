from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..variants import VariantSpec, get_variant


def _group_count(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    )
    values = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([values.sin(), values.cos()], dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class DiffusionResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = x + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class CrossSliceAttention(nn.Module):
    """Attend across ordered slice tokens at each bottleneck location."""

    def __init__(
        self,
        channels: int,
        slices: int,
        heads: int,
        relative_position_encoding: bool,
    ):
        super().__init__()
        self.slices = int(slices)
        token_channels = min(64, max(16, channels // self.slices))
        while token_channels % int(heads):
            token_channels += 1
        self.token_channels = token_channels
        self.to_tokens = nn.Conv2d(channels, self.slices * token_channels, kernel_size=1)
        if relative_position_encoding:
            self.position = nn.Parameter(torch.zeros(1, self.slices, token_channels))
            with torch.no_grad():
                coordinates = torch.linspace(-1.0, 1.0, self.slices).view(1, self.slices, 1)
                self.position.copy_(coordinates.expand_as(self.position))
        else:
            self.register_parameter("position", None)
        self.attention = nn.MultiheadAttention(token_channels, int(heads), batch_first=True)
        self.to_output = nn.Conv2d(self.slices * token_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        tokens = self.to_tokens(x).view(
            batch, self.slices, self.token_channels, height, width
        )
        tokens = (
            tokens.permute(0, 3, 4, 1, 2)
            .reshape(batch * height * width, self.slices, self.token_channels)
        )
        if self.position is not None:
            tokens = tokens + self.position
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        attended = (
            attended.view(batch, height, width, self.slices, self.token_channels)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch, self.slices * self.token_channels, height, width)
        )
        return x + self.to_output(attended)


class HCLDiffUNet(nn.Module):
    def __init__(
        self,
        variant: VariantSpec,
        base_channels: int,
        channel_multipliers: list[int],
        attention_heads: int,
        time_embedding_multiplier: int,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.variant = variant
        self.slices = int(variant.slices)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        channels = [int(base_channels) * int(value) for value in channel_multipliers]
        if not channels:
            raise ValueError("channel_multipliers must not be empty")
        time_channels = int(base_channels) * int(time_embedding_multiplier)
        self.time_mlp = nn.Sequential(
            nn.Linear(int(base_channels), time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input_conv = nn.Conv2d(4 * self.slices, channels[0], kernel_size=3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        for index, channel in enumerate(channels):
            self.encoder_blocks.append(
                nn.ModuleList(
                    [
                        DiffusionResidualBlock(current, channel, time_channels),
                        DiffusionResidualBlock(channel, channel, time_channels),
                    ]
                )
            )
            current = channel
            if index < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv2d(channel, channels[index + 1], kernel_size=3, stride=2, padding=1)
                )
                current = channels[index + 1]

        bottleneck_channels = channels[-1] * 2
        self.middle1 = DiffusionResidualBlock(channels[-1], bottleneck_channels, time_channels)
        self.cross_slice = (
            CrossSliceAttention(
                channels=bottleneck_channels,
                slices=self.slices,
                heads=int(attention_heads),
                relative_position_encoding=variant.relative_slice_encoding,
            )
            if variant.cross_slice_attention
            else nn.Identity()
        )
        self.middle2 = DiffusionResidualBlock(
            bottleneck_channels, bottleneck_channels, time_channels
        )

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current = bottleneck_channels
        for channel in reversed(channels):
            self.upsamples.append(nn.ConvTranspose2d(current, channel, kernel_size=2, stride=2))
            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        DiffusionResidualBlock(channel * 2, channel, time_channels),
                        DiffusionResidualBlock(channel, channel, time_channels),
                    ]
                )
            )
            current = channel
        self.output_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.output = nn.Conv2d(channels[0], self.slices, kernel_size=3, padding=1)

    def _apply_block(
        self,
        block: DiffusionResidualBlock,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                block, x, time_embedding, use_reentrant=False
            )
        return block(x, time_embedding)

    def forward(
        self,
        noisy_ct: torch.Tensor,
        masked_ct: torch.Tensor,
        target_mask: torch.Tensor,
        edit_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(
            sinusoidal_embedding(timesteps, self.input_conv.out_channels)
        )
        x = self.input_conv(
            torch.cat([noisy_ct, masked_ct, target_mask, edit_mask], dim=1)
        )
        skips: list[torch.Tensor] = []
        for index, blocks in enumerate(self.encoder_blocks):
            x = self._apply_block(blocks[0], x, time_embedding)
            x = self._apply_block(blocks[1], x, time_embedding)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        x = self._apply_block(self.middle1, x, time_embedding)
        x = self.cross_slice(x)
        x = self._apply_block(self.middle2, x, time_embedding)
        for upsample, blocks, skip in zip(
            self.upsamples, self.decoder_blocks, reversed(skips), strict=True
        ):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self._apply_block(blocks[0], x, time_embedding)
            x = self._apply_block(blocks[1], x, time_embedding)
        return self.output(F.silu(self.output_norm(x)))


def build_diffusion_model(config: dict[str, Any]) -> HCLDiffUNet:
    model = config["model"]
    variant = get_variant(model["variant"])
    return HCLDiffUNet(
        variant=variant,
        base_channels=int(model["base_channels"]),
        channel_multipliers=[int(value) for value in model["channel_multipliers"]],
        attention_heads=int(model["attention_heads"]),
        time_embedding_multiplier=int(model["time_embedding_multiplier"]),
        gradient_checkpointing=bool(model["gradient_checkpointing"]),
    )
