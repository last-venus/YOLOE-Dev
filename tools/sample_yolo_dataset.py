#!/usr/bin/env python3
"""从 YOLO 格式数据集中抽样，并切分成互不重叠的 train/val/test。"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLITS = ("train", "val", "test")

# 运行前直接修改这里的配置。
CONFIG = {
    "data_root": Path("/home/jingxiuya/datasets/african-wildlife-seg"),
    "output_root": Path("/home/jingxiuya/datasets/african-wildlife-seg-overfit12"),
    "yaml_path": None,  # 设为 None 时，默认输出到 <output_root>/<output_root_name>.yaml
    "source_split": "train",  # 可选：train / val / test
    "samples_per_class": 10,  # 每个类别尽量抽取这么多张不重复图片
    "max_images": None,  # 可选：对最终抽样总数加一个上限
    "split_ratios": {"train": 0.6, "val": 0.3, "test": 0.1},  # 抽样后按比例做互斥切分
    "seed": 0,
    "overwrite": True,
}


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def find_dataset_yaml(data_root: Path) -> Path:
    preferred = data_root / f"{data_root.name}.yaml"
    if preferred.exists():
        return preferred
    yaml_files = sorted(data_root.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"在 {data_root} 下没有找到数据集 YAML 文件")
    return yaml_files[0]


def load_dataset_meta(data_root: Path) -> tuple[Path, dict]:
    yaml_path = find_dataset_yaml(data_root)
    with open(yaml_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    return yaml_path, meta


def normalize_names(names: dict | list) -> dict[int, str]:
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in dict(names).items()}


def resolve_split_entry(data_root: Path, dataset_path: Path, entry: str | list[str]) -> list[Path]:
    items = entry if isinstance(entry, list) else [entry]
    resolved: list[Path] = []
    for item in items:
        p = Path(item)
        if not p.is_absolute():
            p = dataset_path / p
        if not p.exists():
            p = data_root / item
        if not p.exists():
            raise FileNotFoundError(f"无法解析 split 路径配置：{item}")
        resolved.append(p)
    return resolved


def iter_images(entries: list[Path]) -> list[Path]:
    images: list[Path] = []
    for entry in entries:
        if entry.is_dir():
            images.extend(sorted(p for p in entry.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            for line in entry.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = (entry.parent / p).resolve()
                images.append(p)
    return images


def image_to_label_path(image_path: Path) -> Path:
    image_dir = image_path.parent
    if image_dir.name not in DEFAULT_SPLITS:
        raise ValueError(f"图片路径里没有识别到合法的 split 目录：{image_path}")
    label_dir = image_dir.parents[1] / "labels" / image_dir.name
    return label_dir / f"{image_path.stem}.txt"


def parse_label_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        classes.add(int(float(parts[0])))
    return classes


def collect_candidates(image_paths: list[Path]) -> tuple[list[dict], dict[int, list[dict]]]:
    candidates: list[dict] = []
    by_class: dict[int, list[dict]] = defaultdict(list)
    for image_path in image_paths:
        label_path = image_to_label_path(image_path)
        if not label_path.exists():
            continue
        classes = parse_label_classes(label_path)
        if not classes:
            continue
        record = {"image": image_path, "label": label_path, "classes": classes}
        candidates.append(record)
        for cls in classes:
            by_class[cls].append(record)
    return candidates, by_class


def select_records(
    names: dict[int, str],
    candidates: list[dict],
    by_class: dict[int, list[dict]],
    samples_per_class: int,
    max_images: int | None,
    seed: int,
) -> tuple[list[dict], dict[str, list[str]]]:
    rng = random.Random(seed)
    selected: list[dict] = []
    selected_stems: set[str] = set()
    selected_by_class: dict[str, list[str]] = {}

    for cls_idx, cls_name in names.items():
        pool = list(by_class.get(cls_idx, []))
        rng.shuffle(pool)
        picked: list[str] = []
        for record in pool:
            stem = record["image"].stem
            if stem in selected_stems:
                continue
            selected.append(record)
            selected_stems.add(stem)
            picked.append(stem)
            if len(picked) >= samples_per_class:
                break
        selected_by_class[cls_name] = picked

    if max_images is not None and len(selected) > max_images:
        rng.shuffle(selected)
        selected = selected[:max_images]
        selected_stems = {record["image"].stem for record in selected}
        selected_by_class = {
            cls_name: [stem for stem in stems if stem in selected_stems] for cls_name, stems in selected_by_class.items()
        }

    selected.sort(key=lambda record: record["image"].name)
    return selected, selected_by_class


def normalize_split_ratios(raw_ratios: dict[str, float] | None) -> dict[str, float]:
    ratios = {split: 0.0 for split in DEFAULT_SPLITS}
    if raw_ratios is None:
        raise ValueError("CONFIG['split_ratios'] 不能为空")

    unknown = set(raw_ratios) - set(DEFAULT_SPLITS)
    if unknown:
        raise ValueError(f"split_ratios 包含未知 split：{sorted(unknown)}")

    for split in DEFAULT_SPLITS:
        value = float(raw_ratios.get(split, 0.0))
        if value < 0:
            raise ValueError(f"split_ratios['{split}'] 不能小于 0，当前值为：{value}")
        ratios[split] = value

    if sum(ratios.values()) <= 0:
        raise ValueError("split_ratios 的总和必须大于 0")
    return ratios


def allocate_split_counts(total: int, split_ratios: dict[str, float]) -> dict[str, int]:
    if total < 1:
        return {split: 0 for split in DEFAULT_SPLITS}

    ratio_sum = sum(split_ratios.values())
    raw_counts = {split: total * split_ratios[split] / ratio_sum for split in DEFAULT_SPLITS}
    counts = {split: int(raw_counts[split]) for split in DEFAULT_SPLITS}
    remaining = total - sum(counts.values())

    ranked_splits = sorted(
        DEFAULT_SPLITS,
        key=lambda split: (raw_counts[split] - counts[split], split_ratios[split]),
        reverse=True,
    )
    for split in ranked_splits[:remaining]:
        counts[split] += 1

    positive_splits = [split for split in DEFAULT_SPLITS if split_ratios[split] > 0]
    if total >= len(positive_splits):
        for split in positive_splits:
            if counts[split] > 0:
                continue
            donor = max(
                (name for name in positive_splits if counts[name] > 1),
                key=lambda name: counts[name],
                default=None,
            )
            if donor is None:
                break
            counts[donor] -= 1
            counts[split] += 1

    return counts


def split_records(
    records: list[dict],
    split_ratios: dict[str, float],
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    counts = allocate_split_counts(len(shuffled), split_ratios)

    split_records_map: dict[str, list[dict]] = {}
    start = 0
    for split in DEFAULT_SPLITS:
        end = start + counts[split]
        chunk = shuffled[start:end]
        chunk.sort(key=lambda record: record["image"].name)
        split_records_map[split] = chunk
        start = end

    return split_records_map


def copy_split(records: list[dict], output_root: Path, split: str) -> None:
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        shutil.copy2(record["image"], image_dir / record["image"].name)
        shutil.copy2(record["label"], label_dir / record["label"].name)


def maybe_copy_license(data_root: Path, output_root: Path) -> None:
    license_path = data_root / "LICENSE.txt"
    if license_path.exists():
        shutil.copy2(license_path, output_root / license_path.name)


def write_dataset_yaml(output_root: Path, yaml_path: Path, names: dict[int, str]) -> None:
    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_manifest(
    output_root: Path,
    source_yaml: Path,
    source_split: str,
    seed: int,
    split_ratios: dict[str, float],
    selected_by_class: dict[str, list[str]],
    split_records_map: dict[str, list[dict]],
) -> None:
    manifest = {
        "source_yaml": str(source_yaml),
        "source_split": source_split,
        "seed": seed,
        "split_ratios": split_ratios,
        "selected_by_class": selected_by_class,
        "all_selected_images": [
            record["image"].name for split in DEFAULT_SPLITS for record in split_records_map[split]
        ],
        "splits": {
            split: [record["image"].name for record in split_records_map[split]] for split in DEFAULT_SPLITS
        },
    }
    with open(output_root / "selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)


def main() -> None:
    data_root = Path(CONFIG["data_root"]).resolve()
    output_root = Path(CONFIG["output_root"]).resolve()
    yaml_path_cfg = CONFIG.get("yaml_path")
    yaml_path = Path(yaml_path_cfg).resolve() if yaml_path_cfg else (output_root / f"{output_root.name}.yaml").resolve()
    source_split = str(CONFIG.get("source_split", "train"))
    samples_per_class = int(CONFIG.get("samples_per_class", 3))
    max_images = CONFIG.get("max_images")
    split_ratios = normalize_split_ratios(CONFIG.get("split_ratios"))
    seed = int(CONFIG.get("seed", 0))
    overwrite = bool(CONFIG.get("overwrite", False))

    if source_split not in DEFAULT_SPLITS:
        raise ValueError(f"source_split 只能是 {DEFAULT_SPLITS} 之一，当前值为：{source_split}")
    if samples_per_class < 1:
        raise ValueError(f"samples_per_class 必须 >= 1，当前值为：{samples_per_class}")
    if max_images is not None and int(max_images) < 1:
        raise ValueError(f"max_images 必须为 None 或 >= 1，当前值为：{max_images}")
    if max_images is not None:
        max_images = int(max_images)

    if not data_root.is_dir():
        raise FileNotFoundError(f"源数据集目录不存在：{data_root}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在：{output_root}。如需覆盖，请把 CONFIG['overwrite'] 设为 True。")
        remove_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_yaml, meta = load_dataset_meta(data_root)
    names = normalize_names(meta.get("names", {}))
    if not names:
        raise ValueError(f"在 {source_yaml} 中没有找到类别 names 配置")

    dataset_path = Path(meta.get("path", data_root))
    if not dataset_path.is_absolute():
        dataset_path = (source_yaml.parent / dataset_path).resolve()

    split_entries = resolve_split_entry(data_root, dataset_path, meta[source_split])
    image_paths = iter_images(split_entries)
    candidates, by_class = collect_candidates(image_paths)
    if not candidates:
        raise RuntimeError(f"在源 split '{source_split}' 中没有找到带标签的样本")

    records, selected_by_class = select_records(
        names=names,
        candidates=candidates,
        by_class=by_class,
        samples_per_class=samples_per_class,
        max_images=max_images,
        seed=seed,
    )
    if not records:
        raise RuntimeError("抽样结果为空。可以尝试减小 samples_per_class。")

    split_records_map = split_records(records, split_ratios, seed)
    for split in DEFAULT_SPLITS:
        copy_split(split_records_map[split], output_root, split)

    maybe_copy_license(data_root, output_root)
    write_dataset_yaml(output_root, yaml_path, names)
    write_manifest(output_root, source_yaml, source_split, seed, split_ratios, selected_by_class, split_records_map)

    print(f"源数据集：{data_root}")
    print(f"抽样来源 split：{source_split}")
    print(f"输出数据集：{output_root}")
    print(f"数据集 YAML：{yaml_path}")
    print(f"抽中的图片数：{len(records)}")
    print("各 split 图片数：")
    for split in DEFAULT_SPLITS:
        print(f"  {split}: {len(split_records_map[split])} 张")
    for cls_name, stems in selected_by_class.items():
        print(f"  {cls_name}: {len(stems)} 张 -> {stems}")


if __name__ == "__main__":
    main()
