from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3




class TrackerModel(nn.Module):
    """A small CNN baseline for tracker-related classification/regression tasks."""

    def __init__(
        self,
        num_outputs: int = 4,
        hidden_size:int=512
    ) -> None:
        super().__init__()

        self.backbone = efficientnet_b3()
        self.backbone.classifier=nn.Linear(1536,hidden_size)
        self.head = nn.Sequential(

            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size//2, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


def build_tracker_model(
    num_outputs: int = 4,
    hidden_size: int=512
) -> TrackerModel:
    return TrackerModel(
        num_outputs=num_outputs,
        hidden_size=hidden_size
    )


def export_libtorch_script(
    output_path: str | Path,
    model: nn.Module | None = None,
    input_shape: Sequence[int] = (1, 3, 224, 224),
    weights_path: str | Path | None = None,
    device: str = "cpu",
) -> Path:
    """
    Export a TorchScript module that can be loaded directly from libtorch/C++.

    Args:
        output_path: Destination `.pt` path.
        model: Model instance to export. A default TrackerModel is created if omitted.
        input_shape: Example input tensor shape used for tracing.
        weights_path: Optional checkpoint path for `state_dict`.
        device: Export device, usually `cpu` for libtorch deployment.
    """

    export_path = Path(output_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = build_tracker_model()

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


if __name__ == "__main__":
    saved_path = export_libtorch_script("artifacts/tracker_model.pt")
    print(f"Saved TorchScript model to: {saved_path}")
