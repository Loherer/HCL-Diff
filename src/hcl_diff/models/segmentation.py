from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + identity)


class ResUNet2D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, depth: int):
        super().__init__()
        channels = [int(base_channels) * (2**index) for index in range(int(depth))]
        bottleneck_channels = channels[-1] * 2
        self.encoders = nn.ModuleList()
        current = int(in_channels)
        for channel in channels:
            self.encoders.append(ResidualBlock(current, channel))
            current = channel
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ResidualBlock(channels[-1], bottleneck_channels)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = bottleneck_channels
        for channel in reversed(channels):
            self.upsamples.append(nn.ConvTranspose2d(current, channel, kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock(channel * 2, channel))
            current = channel
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True
        ):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)


class SwinUNETR2D(nn.Module):
    def __init__(self, model_config: dict[str, Any], input_channels: int, output_channels: int):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.model = SwinUNETR(
            in_channels=int(input_channels),
            out_channels=int(output_channels),
            patch_size=int(model_config["patch_size"]),
            depths=tuple(int(value) for value in model_config["depths"]),
            num_heads=tuple(int(value) for value in model_config["num_heads"]),
            window_size=int(model_config["window_size"]),
            feature_size=int(model_config["feature_size"]),
            use_checkpoint=bool(model_config["use_checkpoint"]),
            spatial_dims=2,
            use_v2=bool(model_config["use_v2"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _plain_conv_unet(config: dict[str, Any], input_channels: int, output_channels: int) -> nn.Module:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    return PlainConvUNet(
        input_channels=int(input_channels),
        n_stages=int(config["n_stages"]),
        features_per_stage=[int(value) for value in config["features_per_stage"]],
        conv_op=nn.Conv2d,
        kernel_sizes=[int(value) for value in config["kernel_sizes"]],
        strides=[int(value) for value in config["strides"]],
        n_conv_per_stage=[int(value) for value in config["n_conv_per_stage"]],
        num_classes=int(output_channels),
        n_conv_per_stage_decoder=[int(value) for value in config["n_conv_per_stage_decoder"]],
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
        nonlin_first=False,
    )


def build_segmentation_model(config: dict[str, Any]) -> nn.Module:
    model = config["model"]
    backbone = str(model["backbone"]).lower()
    input_channels = int(model["input_channels"])
    output_channels = int(model["output_channels"])
    if backbone == "plainconvunet":
        return _plain_conv_unet(model["plainconvunet"], input_channels, output_channels)
    if backbone == "resunet":
        settings = model["resunet"]
        return ResUNet2D(
            in_channels=input_channels,
            base_channels=int(settings["base_channels"]),
            depth=int(settings["depth"]),
        )
    if backbone == "swinunetr":
        return SwinUNETR2D(model["swinunetr"], input_channels, output_channels)
    raise KeyError(f"Unsupported segmentation backbone: {backbone}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))

