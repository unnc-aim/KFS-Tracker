from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_code.classifier_dataset import build_classifier_dataset
from model_code.classifier_model import build_classifier_model, export_classifier_libtorch_script
from model_code.tracker_dataset import VALID_LABEL_NAMES


@dataclass
class TrainConfig:
    dataset_root: str = "data/coworkers_for_KFS/labeled"
    output_dir: str = "model"
    model_name: str = "tracker_classifier"
    epochs: int = 20
    batch_size: int =50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    image_size: int = 300
    load_checkpoint: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_cache: bool = False


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_image_transform(image_size: int) -> callable:
    def _transform(image: Image.Image) -> torch.Tensor:
        if image_size > 0:
            image = image.resize((image_size, image_size), Image.BILINEAR)
        width, height = image.size
        return (
            torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
            .view(height, width, 3)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

    return _transform


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, int]:
    dataset = build_classifier_dataset(
        root=cfg.dataset_root,
        transform=build_image_transform(cfg.image_size),
        use_cache=cfg.use_cache,
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


def extract_class(batch_target: dict[str, object], device: torch.device) -> torch.Tensor:
    class_id = batch_target["class_id"]
    if not isinstance(class_id, torch.Tensor):
        class_id = torch.as_tensor(class_id, dtype=torch.long)
    return class_id.to(device=device, dtype=torch.long)


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0

    for batch in tqdm(data_loader, desc="val", leave=False):
        images = batch["image"].to(device)
        class_id = extract_class(batch["target"], device)
        logits = model(images)
        loss = criterion(logits, class_id)

        pred_class = logits.argmax(dim=1)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((pred_class == class_id).sum().item())
        total_count += batch_size

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(data_loader, desc="train", leave=False):
        images = batch["image"].to(device)
        class_id = extract_class(batch["target"], device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, class_id)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val: float, path: Path) -> None:
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


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: Path, device: torch.device) -> tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val = float(checkpoint.get("best_val_loss", float("inf")))
    return start_epoch, best_val


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, total_size = build_dataloaders(cfg)
    model = build_classifier_model(num_classes=len(VALID_LABEL_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
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
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch + 1}/{cfg.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        save_checkpoint(model, optimizer, epoch, best_val, last_ckpt)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, best_val, best_ckpt)
            print(f"  New best checkpoint saved: {best_ckpt}")

    script_path = output_dir / f"{cfg.model_name}.pt"
    export_classifier_libtorch_script(
        output_path=script_path,
        model=model,
        weights_path=best_ckpt if best_ckpt.exists() else None,
        input_shape=(1, 3, cfg.image_size, cfg.image_size),
        device=str(device),
    )
    print(f"Training finished. Best val loss={best_val:.6f}. TorchScript exported: {script_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train KFS classifier model.")
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
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--load-checkpoint", type=str, default=TrainConfig.load_checkpoint)
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", help="Enable in-memory image cache for dataset.")
    parser.add_argument("--no-use-cache", dest="use_cache", action="store_false", help="Disable in-memory image cache for dataset.")
    parser.set_defaults(use_cache=TrainConfig.use_cache)
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
        load_checkpoint=args.load_checkpoint,
        device=args.device,
        use_cache=args.use_cache,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)

