from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_code.tracker_dataset import VALID_LABEL_NAMES


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(dataset_root: Path) -> list[Path]:
    images: list[Path] = []
    valid_label_set = set(VALID_LABEL_NAMES)

    for label_dir in sorted(dataset_root.iterdir()):
        if not label_dir.is_dir() or label_dir.name not in valid_label_set:
            continue
        for school_dir in sorted(label_dir.iterdir()):
            if not school_dir.is_dir():
                continue
            for image_path in sorted(school_dir.iterdir()):
                if image_path.is_file() and is_image_file(image_path):
                    images.append(image_path)
    return images


def sample_and_copy(
    dataset_root: Path,
    output_dir: Path,
    ratio: float,
    seed: int,
) -> tuple[int, int]:
    images = collect_images(dataset_root)
    total = len(images)
    if total == 0:
        return 0, 0

    ratio = max(0.0, min(1.0, ratio))
    sample_count = max(1, int(total * ratio))
    sample_count = min(sample_count, total)

    rng = random.Random(seed)
    selected = rng.sample(images, sample_count)

    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for src in selected:
        base_name = src.name
        stem = src.stem
        suffix = src.suffix
        candidate = base_name
        index = 1
        while candidate in used_names or (output_dir / candidate).exists():
            candidate = f"{stem}_{index}{suffix}"
            index += 1
        used_names.add(candidate)
        dst = output_dir / candidate
        shutil.copy2(src, dst)

    return total, sample_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly copy 1%% (or custom ratio) of dataset images into one flat test directory."
    )
    parser.add_argument("--dataset-root", type=str, default="data/coworkers_for_KFS/labeled")
    parser.add_argument("--output-dir", type=str, default="test")
    parser.add_argument("--ratio", type=float, default=0.01, help="Sampling ratio in [0, 1], default=0.01")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    total, copied = sample_and_copy(dataset_root, output_dir, args.ratio, args.seed)
    if total == 0:
        print(f"No images found under dataset root: {dataset_root}")
        return

    print(f"Total images: {total}")
    print(f"Copied images: {copied}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
