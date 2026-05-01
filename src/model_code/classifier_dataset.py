from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset

from model_code.tracker_dataset import VALID_LABEL_NAMES


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class KFSClassifierDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Image.Image], object] | None = None,
        target_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
        valid_label_names: Iterable[str] = VALID_LABEL_NAMES,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.valid_label_names = list(valid_label_names)
        self.valid_label_name_set = set(valid_label_names)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root}")

        self.samples: list[tuple[Path, int, str, str]] = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No valid classification samples found under: {self.root}")
        self.image_cache: list[Image.Image] = self._build_image_cache()

    def _collect_samples(self) -> list[tuple[Path, int, str, str]]:
        out: list[tuple[Path, int, str, str]] = []
        for label_dir in sorted(self.root.iterdir()):
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            if label_name not in self.valid_label_name_set:
                continue
            class_id = self.valid_label_names.index(label_name)

            for school_dir in sorted(label_dir.iterdir()):
                if not school_dir.is_dir():
                    continue
                school_name = school_dir.name
                for image_path in sorted(school_dir.iterdir()):
                    if not image_path.is_file() or not _is_image_file(image_path):
                        continue
                    out.append((image_path, class_id, label_name, school_name))
        return out

    def _build_image_cache(self) -> list[Image.Image]:
        cache: list[Image.Image] = []
        for image_path, _, _, _ in self.samples:
            image = Image.open(image_path).convert("RGB")
            cache.append(image.copy())
            image.close()
        return cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, class_id, label_name, school_name = self.samples[index]
        image = self.image_cache[index]

        if self.transform is not None:
            image_out = self.transform(image)
        else:
            width, height = image.size
            image_out = (
                torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
                .view(height, width, 3)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

        target: dict[str, object] = {
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "label_name": label_name,
            "school_name": school_name,
            "image_path": str(image_path),
        }
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"image": image_out, "target": target}


def build_classifier_dataset(
    root: str | Path = "data/coworkers_for_KFS/labeled",
    transform: Callable[[Image.Image], object] | None = None,
    target_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> KFSClassifierDataset:
    return KFSClassifierDataset(
        root=root,
        transform=transform,
        target_transform=target_transform,
    )
