#!/usr/bin/env python3
"""Generate an African-wildlife segmentation dataset under /home/jingxiuya/datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS_ROOT = ROOT.parent / "datasets"
DEFAULT_DATA_ROOT = DATASETS_ROOT / "african-wildlife"
DEFAULT_OUTPUT_ROOT = DATASETS_ROOT / "african-wildlife-seg"
DEFAULT_YAML_PATH = DATASETS_ROOT / "african-wildlife-seg.yaml"
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_NAMES = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM/SAM2 to convert the African-wildlife detection dataset into a segmentation dataset."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Source dataset root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Generated dataset root.")
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=DEFAULT_YAML_PATH,
        help="Generated dataset YAML path.",
    )
    parser.add_argument("--sam-model", type=str, default="sam2.1_b.pt", help="SAM/SAM2 checkpoint name or path.")
    parser.add_argument("--device", type=str, default="", help="Inference device, e.g. 0, cuda:0, cpu.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
        help="Dataset splits to convert.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How images are placed into the generated dataset root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files if they already exist.")
    return parser.parse_args()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def find_dataset_yaml(data_root: Path) -> Path | None:
    preferred = data_root / f"{data_root.name}.yaml"
    if preferred.exists():
        return preferred
    yaml_files = sorted(data_root.glob("*.yaml"))
    return yaml_files[0] if yaml_files else None


def load_dataset_meta(data_root: Path) -> dict:
    meta = {}
    yaml_path = find_dataset_yaml(data_root)
    if yaml_path:
        with open(yaml_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}

    names = meta.get("names", DEFAULT_NAMES)
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    else:
        names = {int(k): v for k, v in dict(names).items()}

    return {
        "names": names,
        "train": meta.get("train", "images/train"),
        "val": meta.get("val", "images/val"),
        "test": meta.get("test", "images/test"),
    }


def validate_split_dirs(data_root: Path, splits: list[str]) -> None:
    for split in splits:
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")


def link_or_copy_images(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        remove_path(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def convert_split(data_root: Path, output_root: Path, split: str, sam_model: str, device: str, overwrite: bool) -> None:
    try:
        from ultralytics.data.converter import yolo_bbox2segment
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Ultralytics SAM conversion dependencies. "
            "Please check that numpy/opencv/torch in the current environment are compatible."
        ) from exc

    output_label_dir = output_root / "labels" / split
    if output_label_dir.exists():
        if overwrite:
            remove_path(output_label_dir)
        else:
            print(f"[skip] labels/{split} already exists: {output_label_dir}")
            return

    print(f"[convert] {split}: {data_root / 'images' / split} -> {output_label_dir}")
    yolo_bbox2segment(
        im_dir=data_root / "images" / split,
        save_dir=output_label_dir,
        sam_model=sam_model,
        device=device or None,
    )


def copy_license(data_root: Path, output_root: Path, overwrite: bool) -> None:
    license_path = data_root / "LICENSE.txt"
    if not license_path.exists():
        return

    dst = output_root / license_path.name
    if dst.exists():
        if not overwrite:
            return
        remove_path(dst)
    shutil.copy2(license_path, dst)


def write_dataset_yaml(output_root: Path, yaml_path: Path, meta: dict) -> Path:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(output_root.resolve()),
        "train": meta["train"],
        "val": meta["val"],
        "test": meta["test"],
        "names": meta["names"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def main(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    yaml_path = args.yaml_path.resolve()
    splits = list(dict.fromkeys(args.splits))

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    validate_split_dirs(data_root, splits)
    meta = load_dataset_meta(data_root)

    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "labels").mkdir(parents=True, exist_ok=True)

    for split in splits:
        link_or_copy_images(
            src=data_root / "images" / split,
            dst=output_root / "images" / split,
            mode=args.image_mode,
            overwrite=args.overwrite,
        )
        convert_split(
            data_root=data_root,
            output_root=output_root,
            split=split,
            sam_model=args.sam_model,
            device=args.device,
            overwrite=args.overwrite,
        )

    copy_license(data_root, output_root, args.overwrite)
    yaml_path = write_dataset_yaml(output_root, yaml_path, meta)

    print("Done.")
    print(f"Segmentation dataset root: {output_root}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Train with: yolo segment train data={yaml_path} model=yolo11n-seg.pt imgsz=640")


if __name__ == "__main__":
    main(parse_args())
