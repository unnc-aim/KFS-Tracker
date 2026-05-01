from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class HeatmapModel(nn.Module):
    def __init__(self, in_channels: int = 3, heatmap_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.in_conv = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.bottleneck = Down(c4, c4)

        self.up1 = Up(c4, c4, c3)
        self.up2 = Up(c3, c3, c2)
        self.up3 = Up(c2, c2, c1)
        self.up4 = Up(c1, c1, c1)
        self.heatmap_head = nn.Conv2d(c1, heatmap_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bottleneck(x4)

        x = self.up1(xb, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        heatmaps = self.heatmap_head(x)
        return heatmaps


def build_heatmap_model(
    in_channels: int = 3,
    heatmap_channels: int = 4,
    base_channels: int = 32,
) -> HeatmapModel:
    return HeatmapModel(
        in_channels=in_channels,
        heatmap_channels=heatmap_channels,
        base_channels=base_channels,
    )


def export_heatmap_libtorch_script(
    output_path: str | Path,
    model: nn.Module | None = None,
    input_shape: Sequence[int] = (1, 3, 300, 300),
    weights_path: str | Path | None = None,
    device: str = "cpu",
) -> Path:
    export_path = Path(output_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = build_heatmap_model()

    target_device = torch.device(device)
    model = model.to(target_device)

    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=target_device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    model.eval()
    example_input = torch.randn(*input_shape, device=target_device)
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, example_input)
        scripted_model = torch.jit.freeze(scripted_model)
        scripted_model.save(str(export_path))

    return export_path
