from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class ClassifierModel(nn.Module):
    def __init__(self, num_classes: int = 32) -> None:
        super().__init__()
        self.backbone = efficientnet_b0()
        self.backbone.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_classifier_model(num_classes: int = 32) -> ClassifierModel:
    return ClassifierModel(num_classes=num_classes)


def export_classifier_libtorch_script(
    output_path: str | Path,
    model: nn.Module | None = None,
    input_shape: Sequence[int] = (1, 3, 300, 300),
    weights_path: str | Path | None = None,
    device: str = "cpu",
) -> Path:
    export_path = Path(output_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = build_classifier_model()

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

