from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
from PIL import Image
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_code.heatmap_dataset import build_heatmap_dataset
from model_code.heatmap_model import build_heatmap_model, export_heatmap_libtorch_script


@dataclass
class TrainConfig:
    dataset_root: str = "data/coworkers_for_KFS/labeled"
    output_dir: str = "model"
    model_name: str = "corner_heatmap_localizer"
    epochs: int = 30
    batch_size: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    image_size: int = 300
    heatmap_size: int = 300
    heatmap_sigma: float = 3.0
    base_channels: int = 64
    load_checkpoint: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_image_dir: str="test"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_image_transform(image_size: int) -> callable:
    def _transform(image: Image.Image) -> torch.Tensor:
        if image_size > 0:
            image = image.resize((image_size, image_size), Image.BILINEAR)
        width, height = image.size
        tensor = (
            torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
            .view(height, width, 3)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        return tensor

    return _transform


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, int]:
    dataset = build_heatmap_dataset(
        root=cfg.dataset_root,
        transform=build_image_transform(cfg.image_size),
        image_size=cfg.image_size,
        heatmap_size=cfg.heatmap_size,
        heatmap_sigma=cfg.heatmap_sigma,
    )

    total_size = len(dataset)
    val_size = max(1, int(total_size * cfg.val_ratio))
    train_size = total_size - val_size
    if train_size <= 0:
        raise RuntimeError(f"Dataset too small for split: total={total_size}, val={val_size}")

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, total_size


def extract_heatmaps(batch_target: dict[str, object], device: torch.device) -> torch.Tensor:
    heatmaps = batch_target["heatmaps"]
    if not isinstance(heatmaps, torch.Tensor):
        heatmaps = torch.as_tensor(heatmaps, dtype=torch.float32)
    return heatmaps.to(device=device, dtype=torch.float32)


def decode_corners_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    b, c, h, w = heatmaps.shape
    flat = heatmaps.view(b, c, -1)
    idx = flat.argmax(dim=-1)
    ys = (idx // w).float()
    xs = (idx % w).float()
    x_norm = xs / max(w - 1, 1)
    y_norm = ys / max(h - 1, 1)
    corners = torch.stack([x_norm, y_norm], dim=-1).view(b, c * 2)
    return corners


def order_points_tl_bl_br_tr(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0])]
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    tl = tuple(left[0].astype(np.int32))
    bl = tuple(left[1].astype(np.int32))
    tr = tuple(right[0].astype(np.int32))
    br = tuple(right[1].astype(np.int32))
    return [tl, bl, br, tr]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    heatmap_criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(data_loader, desc="val", leave=False):
        images = batch["image"].to(device)
        target_heatmaps = extract_heatmaps(batch["target"], device)

        pred_heatmaps = model(images)
        heatmap_loss = heatmap_criterion(pred_heatmaps, target_heatmaps)
        loss = heatmap_loss

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    heatmap_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(data_loader, desc="train", leave=False):
        images = batch["image"].to(device)
        target_heatmaps = extract_heatmaps(batch["target"], device)

        optimizer.zero_grad(set_to_none=True)
        pred_heatmaps = model(images)
        heatmap_loss = heatmap_criterion(pred_heatmaps, target_heatmaps)
        loss = heatmap_loss
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val = float(checkpoint.get("best_val_loss", float("inf")))
    return start_epoch, best_val


@torch.no_grad()
def visualization(
    image_dir: str | Path,
    output_path: str | Path = "heatmap_vis.jpg",
    checkpoint_path: str | Path = "model/corner_heatmap_localizer_best.pth",
    image_size: int = 300,
    heatmap_size: int = 300,
    base_channels: int = 64,
    device: str | torch.device | None = None,
) -> Path:
    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_candidates: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_candidates.extend(image_dir.rglob(ext))
    if not image_candidates:
        raise RuntimeError(f"No image files found in directory: {image_dir}")

    image_path = random.choice(image_candidates)

    device_obj = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_heatmap_model(heatmap_channels=4, base_channels=base_channels).to(device_obj)

    ckpt = torch.load(Path(checkpoint_path), map_location=device_obj)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None or image_bgr.size == 0:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize((image_size, image_size), Image.BILINEAR)
    image_tensor = (
        torch.tensor(bytearray(image_pil.tobytes()), dtype=torch.uint8)
        .view(image_size, image_size, 3)
        .permute(2, 0, 1)
        .float()
        / 255.0
    ).unsqueeze(0).to(device_obj)

    pred_heatmaps = model(image_tensor)
    pred_heatmaps = torch.sigmoid(pred_heatmaps)
    pred_corners_norm = decode_corners_from_heatmaps(pred_heatmaps).squeeze(0).detach().cpu().numpy()

    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    points: list[tuple[int, int]] = []
    for i in range(4):
        x = max(0, min(int(round(float(pred_corners_norm[2 * i]) * (w - 1))), w - 1))
        y = max(0, min(int(round(float(pred_corners_norm[2 * i + 1]) * (h - 1))), h - 1))
        points.append((x, y))
    points = order_points_tl_bl_br_tr(points)

    for i, p in enumerate(points):
        cv2.circle(annotated, p, 5, (0, 255, 255), -1)
        cv2.putText(annotated, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    polygon = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [polygon], True, (0, 255, 0), 2)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    return out_path


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, total_size = build_dataloaders(cfg)
    model = build_heatmap_model(heatmap_channels=4, base_channels=cfg.base_channels).to(device)
    heatmap_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    best_val = float("inf")
    if cfg.load_checkpoint:
        ckpt_path = Path(cfg.load_checkpoint)
        if ckpt_path.exists():
            start_epoch, best_val = load_checkpoint(model, optimizer, ckpt_path, device)
            print(f"Loaded checkpoint: {ckpt_path}, start_epoch={start_epoch}, best_val={best_val:.6f}")
        else:
            print(f"Checkpoint not found, training from scratch: {ckpt_path}")

    best_ckpt = output_dir / f"{cfg.model_name}_best.pth"
    last_ckpt = output_dir / f"{cfg.model_name}_last.pth"
    print(
        f"Start training: total={total_size}, train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
        f"device={device}, epochs={cfg.epochs}"
    )

    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, heatmap_criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, heatmap_criterion, device)
        print(f"[Epoch {epoch + 1}/{cfg.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        save_checkpoint(model, optimizer, epoch, best_val, last_ckpt)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, best_val, best_ckpt)
            try:
                visualization(
                    image_dir=cfg.test_image_dir,
                    base_channels=cfg.base_channels,
                    image_size=cfg.image_size,
                    heatmap_size=cfg.heatmap_size,
                    device=cfg.device,
                    checkpoint_path=best_ckpt,
                    output_path=output_dir / f"{cfg.model_name}_vis.jpg",
                )
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  Visualization skipped: {e}")
            except Exception as e:
                print("Visualization Error: ",e)
            print(f"  New best checkpoint saved: {best_ckpt}")

    script_path = output_dir / f"{cfg.model_name}.pt"
    export_heatmap_libtorch_script(
        output_path=script_path,
        model=model,
        weights_path=best_ckpt if best_ckpt.exists() else None,
        input_shape=(1, 3, cfg.image_size, cfg.image_size),
        device=str(device),
    )
    print(f"Training finished. Best val loss={best_val:.6f}. TorchScript exported: {script_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train KFS heatmap corner localization model.")
    parser.add_argument("--dataset-root", type=str, default=TrainConfig.dataset_root)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--model-name", type=str, default=TrainConfig.model_name)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--val-ratio", type=float, default=TrainConfig.val_ratio)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--heatmap-size", type=int, default=TrainConfig.heatmap_size)
    parser.add_argument("--heatmap-sigma", type=float, default=TrainConfig.heatmap_sigma)
    parser.add_argument("--base-channels", type=int, default=TrainConfig.base_channels)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--load-checkpoint", type=str, default=TrainConfig.load_checkpoint)
    parser.add_argument("--test-image-dir",type=str,default=TrainConfig.test_image_dir)
    args = parser.parse_args()

    return TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        base_channels=args.base_channels,
        load_checkpoint=args.load_checkpoint,
        device=args.device,
        test_image_dir=args.test_image_dir,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
