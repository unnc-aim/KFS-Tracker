from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_code.corner_point_dataset import build_dataset, detect_colored_quad_corners
from model_code.tracker_dataset import VALID_LABEL_NAMES
from model_code.tracker_model import build_tracker_model, export_libtorch_script


@dataclass
class TrainConfig:
    dataset_root: str = "data/coworkers_for_KFS/labeled"
    output_dir: str = "model"
    model_name: str = "corner_point_localizer"
    epochs: int = 20
    batch_size: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    image_size: int = 300
    hidden_size: int = 512
    load_checkpoint: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
    transform = build_image_transform(cfg.image_size)
    dataset = build_dataset(root=cfg.dataset_root, transform=transform)

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
        drop_last=True,
    )
    return train_loader, val_loader, total_size


def extract_targets(batch_target: dict[str, object], device: torch.device) -> torch.Tensor:
    corners = batch_target["corners_xy_norm"]
    if not isinstance(corners, torch.Tensor):
        corners = torch.as_tensor(corners, dtype=torch.float32)
    return corners.to(device=device, dtype=torch.float32)


def extract_class(batch_target: dict[str, object], device: torch.device) -> torch.Tensor:
    class_id = batch_target["class_id"]
    if not isinstance(class_id, torch.Tensor):
        class_id = torch.as_tensor(class_id, dtype=torch.long)
    return class_id.to(device=device, dtype=torch.long)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    point_criterion: nn.Module,
    classification_criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0

    for batch in tqdm(data_loader, desc="val", leave=False):
        images = batch["image"].to(device)
        targets = extract_targets(batch["target"], device)
        class_id = extract_class(batch["target"], device)
        preds, class_id_pred = model(images)
        loss = point_criterion(preds, targets)
        pred_class = class_id_pred.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((pred_class == class_id).sum().item())
        total_count += batch_size

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    
    # traditional_visualization("data/coworkers_for_KFS/labeled/F_25/WTR/frame_F_25_0009.jpg")
    return avg_loss, avg_acc


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    point_criterion: nn.Module,
    classification_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    running_point_loss = 0.0
    running_cls_loss = 0.0
    running_steps = 0

    for batch in tqdm(data_loader, desc="train", leave=False):
        images = batch["image"].to(device)
        targets = extract_targets(batch["target"], device)
        class_id = extract_class(batch["target"], device)

        optimizer.zero_grad(set_to_none=True)
        preds, class_id_pred = model(images)
        point_loss = point_criterion(preds, targets)
        cls_loss = classification_criterion(class_id_pred, class_id)

        # running_point_loss += float(point_loss.item())
        # running_cls_loss += float(cls_loss.item())
        # running_steps += 1

        # avg_point_loss = running_point_loss / max(running_steps, 1)
        # avg_cls_loss = running_cls_loss / max(running_steps, 1)
        # point_weight = 1.0 / max(avg_point_loss, 1e-6)
        # cls_weight = 1.0 / max(avg_cls_loss, 1e-6)

        # loss = point_weight * point_loss + cls_weight * cls_loss
        loss=point_loss+cls_loss
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
    image_path: str | Path,
    output_path: str | Path = "corner_point_vis.jpg",
    checkpoint_path: str | Path = "model/corner_point_localizer_best.pth",
    image_size: int = 300,
    hidden_size: int = 512,
    device: str | torch.device | None = None,
) -> Path:
    device_obj = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_tracker_model(num_outputs=8, hidden_size=hidden_size).to(device_obj)

    ckpt = torch.load(Path(checkpoint_path), map_location=device_obj)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    image_path = Path(image_path)
    output_path = Path(output_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None or image_bgr.size == 0:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    transform = build_image_transform(image_size)
    image_tensor = transform(image_pil).unsqueeze(0).to(device_obj)

    corners_pred, class_logits = model(image_tensor)
    corners_pred = corners_pred.squeeze(0).detach().cpu().float().clamp(0.0, 1.0)
    class_probs = torch.softmax(class_logits.squeeze(0).detach().cpu(), dim=0)
    class_id = int(class_probs.argmax().item())
    class_score = float(class_probs[class_id].item())
    class_name = VALID_LABEL_NAMES[class_id] if 0 <= class_id < len(VALID_LABEL_NAMES) else str(class_id)

    height, width = image_bgr.shape[:2]
    points: list[tuple[int, int]] = []
    for i in range(4):
        x_norm = float(corners_pred[2 * i].item())
        y_norm = float(corners_pred[2 * i + 1].item())
        x = max(0, min(int(round(x_norm * width)), width - 1))
        y = max(0, min(int(round(y_norm * height)), height - 1))
        points.append((x, y))

    annotated = image_bgr.copy()
    for i, pt in enumerate(points):
        cv2.circle(annotated, pt, 5, (0, 255, 255), -1)
        cv2.putText(
            annotated,
            str(i),
            (pt[0] + 6, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    polygon = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [polygon], True, (0, 255, 0), 2)
    cv2.putText(
        annotated,
        f"class: {class_name} (id={class_id}, p={class_score:.3f})",
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (50, 220, 50),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return output_path


def traditional_visualization(
    image_path: str | Path,
    output_path: str | Path = "corner_point_traditional_vis.jpg",
    color: str | None = None,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None or image_bgr.size == 0:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    found, corners_xy_norm = detect_colored_quad_corners(image_bgr, color)
    annotated = image_bgr.copy()

    if found and corners_xy_norm is not None and len(corners_xy_norm) >= 8:
        height, width = image_bgr.shape[:2]
        points: list[tuple[int, int]] = []
        for i in range(4):
            x_norm = float(corners_xy_norm[2 * i])
            y_norm = float(corners_xy_norm[2 * i + 1])
            x = max(0, min(int(round(x_norm * width)), width - 1))
            y = max(0, min(int(round(y_norm * height)), height - 1))
            points.append((x, y))

        polygon = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [polygon], True, (0, 255, 0), 2)
        for i, pt in enumerate(points):
            cv2.circle(annotated, pt, 5, (0, 255, 255), -1)
            cv2.putText(
                annotated,
                str(i),
                (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(annotated, "traditional detect: true", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)
    else:
        cv2.putText(annotated, "traditional detect: false", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return output_path


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, total_size = build_dataloaders(cfg)

    model = build_tracker_model(num_outputs=8, hidden_size=cfg.hidden_size).to(device)
    point_criterion = nn.SmoothL1Loss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    best_val = float("inf")
    best_acc=0
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
        train_loss = train_one_epoch(
            model,
            train_loader,
            point_criterion,
            classification_criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            point_criterion,
            classification_criterion,
            device,
        )

        print(f"[Epoch {epoch + 1}/{cfg.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")

        save_checkpoint(model, optimizer, epoch, best_val, last_ckpt)
        if val_loss <= best_val and val_acc>=best_acc:
            best_val = val_loss
            best_acc=val_acc
            save_checkpoint(model, optimizer, epoch, best_val, best_ckpt)
            print(f"  New best checkpoint saved: {best_ckpt}")
            visualization("data/coworkers_for_KFS/labeled/F_25/WTR/frame_F_25_0009.jpg")

    script_path = output_dir / f"{cfg.model_name}.pt"
    export_libtorch_script(
        output_path=script_path,
        model=model,
        weights_path=best_ckpt if best_ckpt.exists() else None,
        input_shape=(1, 3, cfg.image_size, cfg.image_size),
        device=str(device),
    )
    print(f"Training finished. Best val loss={best_val:.6f}. TorchScript exported: {script_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train KFS corner-point localization model.")
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
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--load-checkpoint", type=str, default=TrainConfig.load_checkpoint)
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
        hidden_size=args.hidden_size,
        load_checkpoint=args.load_checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
    
