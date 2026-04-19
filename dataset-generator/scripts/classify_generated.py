from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


LABEL_PATTERN = re.compile(r"^gen_(?P<label>.+?)_sid\d+_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 data/generated 下的图片与数据按标签归类到同一目录，"
            "目录结构为: <标签>/<学校或战队名>/frame_xxxxxx.{jpg,txt}"
        )
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("data/generated"),
        help="generated 根目录（默认: data/generated）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/generated/classified"),
        help="归类后的输出目录（默认: data/generated/classified）",
    )
    parser.add_argument(
        "--team-name",
        type=str,
        default=None,
        help="学校缩写/战队英文名（如 UNNC、MTI）；不传则交互输入",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="文件处理模式：copy=复制（默认），move=移动",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览不实际写入",
    )
    return parser.parse_args()


def sanitize_team_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "", (name or "").strip())
    if not cleaned:
        raise ValueError("队名不能为空，且仅允许字母/数字/_/-")
    return cleaned


def extract_label(stem: str) -> str | None:
    match = LABEL_PATTERN.match(stem)
    return match.group("label") if match else None


def collect_pairs(images_dir: Path, data_dir: Path) -> list[tuple[Path, Path, str]]:
    image_files = [p for p in images_dir.iterdir() if p.is_file()]
    image_files.sort(key=lambda p: p.name)

    pairs: list[tuple[Path, Path, str]] = []
    for image_path in image_files:
        data_path = data_dir / f"{image_path.stem}.txt"
        if not data_path.exists():
            continue

        label = extract_label(image_path.stem)
        if label is None:
            continue

        pairs.append((image_path, data_path, label))

    return pairs


def transfer(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def run(
    generated_root: Path,
    output_root: Path,
    team_name: str,
    mode: str = "copy",
    dry_run: bool = False,
) -> None:
    images_dir = generated_root / "images"
    data_dir = generated_root / "data"

    if not images_dir.exists() or not data_dir.exists():
        raise FileNotFoundError(
            f"未找到目录: {images_dir} 或 {data_dir}，请确认 generated 路径是否正确。"
        )

    pairs = collect_pairs(images_dir, data_dir)
    if not pairs:
        print("未找到可归类的图片-数据配对文件。")
        return

    grouped: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for image_path, data_path, label in pairs:
        grouped[label].append((image_path, data_path))

    total = 0
    print(f"\n准备归类，共匹配到 {len(pairs)} 对文件，标签数 {len(grouped)}。")
    print(f"输出目录: {output_root.resolve()}")
    print(f"模式: {mode}{'（预览）' if dry_run else ''}\n")

    for label in sorted(grouped):
        entries = sorted(grouped[label], key=lambda x: x[0].name)
        target_dir = output_root / label / team_name
        print(f"[{label}] -> {target_dir}，共 {len(entries)} 对")

        for idx, (image_path, data_path) in enumerate(entries, start=1):
            frame_base = f"frame_{idx:06d}"
            target_image = target_dir / f"{frame_base}{image_path.suffix.lower()}"
            target_data = target_dir / f"{frame_base}.txt"

            if dry_run:
                print(f"  DRY-RUN: {image_path.name} -> {target_image.name}")
                print(f"  DRY-RUN: {data_path.name} -> {target_data.name}")
            else:
                transfer(image_path, target_image, mode=mode)
                transfer(data_path, target_data, mode=mode)

            total += 1

    print(f"\n完成：已处理 {total} 对文件。")


def main() -> None:
    args = parse_args()
    team_name = args.team_name
    if not team_name:
        team_name = input("请输入学校缩写或战队英文名（如 UNNC / MTI）：").strip()

    team_name = sanitize_team_name(team_name)

    try:
        run(
            generated_root=args.generated_root,
            output_root=args.output_root,
            team_name=team_name,
            mode=args.mode,
            dry_run=args.dry_run,
        )
    except BrokenPipeError:
        # 兼容输出被管道截断（例如: | head）
        try:
            sys.stdout.close()
        finally:
            return


if __name__ == "__main__":
    main()
