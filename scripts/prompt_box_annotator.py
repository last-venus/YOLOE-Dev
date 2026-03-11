from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW_NAME = "Prompt Box Annotator"
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900


def get_cv2():
    """延迟导入 cv2，避免只看 --help 或只读 YAML 时也要求 OpenCV 环境完整。"""
    import cv2

    return cv2


def optional_import_yaml():
    """按需导入 PyYAML；如果环境里没有，就退化成 JSON 形式保存。"""
    try:
        import yaml

        return yaml
    except ImportError:
        return None


def choose_images_with_dialog() -> list[Path]:
    """如果用户没有在命令行传图片，就弹出文件选择框。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("当前环境无法使用文件选择框，请改用 --images 或 --image-dir。") from exc

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilenames(
        title="选择要标注的图片",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
    )
    root.destroy()
    return [Path(path) for path in selected]


def collect_images_from_dir(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def read_annotation_file(path: Path) -> dict:
    """读取 YAML；如果没有 PyYAML，就尝试按 JSON 读取。

    这里这样做是为了避免脚本强依赖第三方包，同时依然满足“保存为 yaml 文件”的需求。
    JSON 本身也是 YAML 的子集，所以回退时文件内容依然可以放在 .yaml 里。
    """
    yaml = optional_import_yaml()
    text = path.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"标注文件格式不正确: {path}")
    return data


def write_annotation_file(path: Path, data: dict) -> None:
    """写出 YAML；若无 PyYAML，则写出 JSON 风格的 YAML。"""
    yaml = optional_import_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)

    if yaml is not None:
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference_specs_from_yaml(yaml_path: str | Path) -> list[dict]:
    """供其它脚本 import 使用，把 YAML 转回你当前项目用的格式。"""
    yaml_path = Path(yaml_path)
    data = read_annotation_file(yaml_path)
    reference_specs = []

    for item in data.get("reference_specs", []):
        prompts = item.get("prompts", {})
        reference_specs.append(
            {
                "image": item["image"],
                "prompts": {
                    "bboxes": np.array(prompts.get("bboxes", []), dtype=np.float32),
                    "cls": np.array(prompts.get("cls", []), dtype=np.int32),
                },
            }
        )

    return reference_specs


def build_annotation_data(class_names: list[str], annotations: dict, image_paths: list[Path]) -> dict:
    """把内存里的标注结果整理成统一 YAML 结构。"""
    reference_specs = []
    for image_path in image_paths:
        record = annotations.get(str(image_path), {"bboxes": [], "cls": []})
        reference_specs.append(
            {
                "image": str(image_path),
                "prompts": {
                    "bboxes": [
                        [round(float(value), 1) for value in box]
                        for box in record.get("bboxes", [])
                    ],
                    "cls": [int(value) for value in record.get("cls", [])],
                },
            }
        )

    return {
        "class_names": class_names,
        "reference_specs": reference_specs,
    }


def load_existing_annotations(input_yaml: Path) -> tuple[list[str], dict[str, dict], list[Path]]:
    """从已有 YAML 恢复类别名、标注结果和图片列表。"""
    data = read_annotation_file(input_yaml)
    class_names = list(data.get("class_names", []))
    annotations = {}
    image_paths = []

    for item in data.get("reference_specs", []):
        image_path = Path(item["image"]).expanduser().resolve()
        prompts = item.get("prompts", {})
        annotations[str(image_path)] = {
            "bboxes": [list(map(float, box)) for box in prompts.get("bboxes", [])],
            "cls": [int(value) for value in prompts.get("cls", [])],
        }
        image_paths.append(image_path)

    return class_names, annotations, image_paths


class PromptBoxAnnotator:
    """一个简单的 OpenCV 交互标注器。

    操作说明:
    - 鼠标左键拖拽: 新建一个框
    - 数字键 0~9: 切换当前类别
    - [ / ]: 在类别之间循环切换
    - u 或 Backspace: 撤销当前图片最后一个框
    - c: 清空当前图片的全部框
    - n / Enter / 空格: 下一张图
    - p: 上一张图
    - s: 立即保存 YAML
    - q / ESC: 保存并退出
    """

    def __init__(
        self,
        image_paths: list[Path],
        class_names: list[str],
        output_yaml: Path,
        annotations: dict[str, dict] | None = None,
    ) -> None:
        if not image_paths:
            raise ValueError("没有可标注的图片。")
        if not class_names:
            raise ValueError("至少需要一个类别名。")

        self.image_paths = image_paths
        self.class_names = class_names
        self.output_yaml = output_yaml
        self.annotations = annotations or {}
        self.cv2 = get_cv2()

        self.index = 0
        self.current_class = 0
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.current_image = None
        self.current_display = None
        self.current_scale = 1.0

    def current_image_path(self) -> Path:
        return self.image_paths[self.index]

    def current_record(self) -> dict:
        key = str(self.current_image_path())
        if key not in self.annotations:
            self.annotations[key] = {"bboxes": [], "cls": []}
        return self.annotations[key]

    def load_current_image(self) -> None:
        image_path = self.current_image_path()
        image = self.cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        self.current_image = image
        height, width = image.shape[:2]
        scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
        self.current_scale = scale

        if scale < 1.0:
            display_size = (int(width * scale), int(height * scale))
            self.current_display = self.cv2.resize(
                image,
                display_size,
                interpolation=self.cv2.INTER_AREA,
            )
        else:
            self.current_display = image.copy()

    def image_to_display(self, value: float) -> int:
        return int(round(value * self.current_scale))

    def display_to_image(self, value: int) -> float:
        return float(value / self.current_scale)

    def render(self) -> np.ndarray:
        canvas = self.current_display.copy()
        record = self.current_record()

        for box, cls_id in zip(record["bboxes"], record["cls"]):
            cls_id = int(cls_id)
            x1, y1, x2, y2 = box
            dx1, dy1 = self.image_to_display(x1), self.image_to_display(y1)
            dx2, dy2 = self.image_to_display(x2), self.image_to_display(y2)

            label = f"CLS {cls_id}: {self.class_names[cls_id]}"
            self.cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
            self.cv2.putText(
                canvas,
                label,
                (dx1, max(24, dy1 - 8)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                self.cv2.LINE_AA,
            )

        if self.dragging and self.drag_start and self.drag_end:
            self.cv2.rectangle(canvas, self.drag_start, self.drag_end, (0, 255, 255), 2)

        info_lines = [
            f"Image {self.index + 1}/{len(self.image_paths)}: {self.current_image_path().name}",
            f"Current class: {self.current_class} ({self.class_names[self.current_class]})",
            "Keys: 0-9 class | [ ] switch | u undo | c clear | n next | p prev | s save | q quit",
        ]
        for line_index, text in enumerate(info_lines):
            y = 28 + line_index * 28
            self.cv2.putText(
                canvas,
                text,
                (12, y),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                self.cv2.LINE_AA,
            )

        return canvas

    def save_yaml(self) -> None:
        data = build_annotation_data(self.class_names, self.annotations, self.image_paths)
        write_annotation_file(self.output_yaml, data)
        print(f"已保存标注文件: {self.output_yaml}")

    def add_box(self, start_xy: tuple[int, int], end_xy: tuple[int, int]) -> None:
        x1, y1 = start_xy
        x2, y2 = end_xy
        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            return

        ix1, ix2 = sorted([self.display_to_image(x1), self.display_to_image(x2)])
        iy1, iy2 = sorted([self.display_to_image(y1), self.display_to_image(y2)])

        record = self.current_record()
        record["bboxes"].append([ix1, iy1, ix2, iy2])
        record["cls"].append(self.current_class)
        print(
            f"新增框: image={self.current_image_path().name}, "
            f"cls={self.current_class}, box={[round(v, 1) for v in [ix1, iy1, ix2, iy2]]}"
        )

    def undo_last_box(self) -> None:
        record = self.current_record()
        if record["bboxes"]:
            record["bboxes"].pop()
            record["cls"].pop()
            print(f"已撤销当前图片最后一个框: {self.current_image_path().name}")

    def clear_current_boxes(self) -> None:
        self.annotations[str(self.current_image_path())] = {"bboxes": [], "cls": []}
        print(f"已清空当前图片所有框: {self.current_image_path().name}")

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == self.cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (x, y)
        elif event == self.cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_end = (x, y)
            self.add_box(self.drag_start, self.drag_end)
            self.drag_start = None
            self.drag_end = None

    def next_image(self) -> None:
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.load_current_image()

    def prev_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.load_current_image()

    def run(self) -> None:
        print("标注工具已启动。")
        print("操作: 鼠标左键拖框，数字键切类别，n 下一张，p 上一张，u 撤销，c 清空，s 保存，q 退出。")

        self.load_current_image()
        self.cv2.namedWindow(WINDOW_NAME, self.cv2.WINDOW_NORMAL)
        self.cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            frame = self.render()
            self.cv2.imshow(WINDOW_NAME, frame)
            key = self.cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if key in (27, ord("q")):
                self.save_yaml()
                break
            if key == ord("s"):
                self.save_yaml()
                continue
            if key in (13, 10, 32, ord("n")):
                self.next_image()
                continue
            if key == ord("p"):
                self.prev_image()
                continue
            if key in (8, 127, ord("u")):
                self.undo_last_box()
                continue
            if key == ord("c"):
                self.clear_current_boxes()
                continue
            if key == ord("["):
                self.current_class = (self.current_class - 1) % len(self.class_names)
                continue
            if key == ord("]"):
                self.current_class = (self.current_class + 1) % len(self.class_names)
                continue
            if ord("0") <= key <= ord("9"):
                cls_id = key - ord("0")
                if cls_id < len(self.class_names):
                    self.current_class = cls_id

        self.cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多张参考图 prompt 框标注工具")
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="直接指定若干张图片路径，例如 --images a.jpg b.jpg c.jpg",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="指定一个目录，自动加载其中所有图片",
    )
    parser.add_argument(
        "--input-yaml",
        type=str,
        default="",
        help="读取已有 YAML，继续编辑其中的图片和框",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/jingxiuya/YOLOE/scripts/runs/prompt_annotations/reference_prompts.yaml",
        help="标注结果保存路径",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["paper_cup", "red_cup"],
        help="类别名列表，顺序必须和 cls 编号一致，例如 --class-names paper_cup red_cup",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[list[Path], list[str], dict[str, dict]]:
    """把命令行、目录、已有 YAML 三种输入源合并起来。"""
    image_paths = []
    annotations = {}
    class_names = list(args.class_names)

    if args.input_yaml:
        yaml_class_names, yaml_annotations, yaml_image_paths = load_existing_annotations(Path(args.input_yaml))
        if yaml_class_names and args.class_names == ["paper_cup", "red_cup"]:
            class_names = yaml_class_names
        annotations.update(yaml_annotations)
        image_paths.extend(yaml_image_paths)

    if args.image_dir:
        image_paths.extend(collect_images_from_dir(Path(args.image_dir)))

    if args.images:
        image_paths.extend(Path(path) for path in args.images)

    if not image_paths:
        image_paths = choose_images_with_dialog()

    image_paths = unique_paths(image_paths)
    if not image_paths:
        raise ValueError("没有选中任何图片。")

    return image_paths, class_names, annotations


def main() -> None:
    args = parse_args()
    image_paths, class_names, annotations = resolve_inputs(args)

    annotator = PromptBoxAnnotator(
        image_paths=image_paths,
        class_names=class_names,
        output_yaml=Path(args.output).expanduser().resolve(),
        annotations=annotations,
    )
    annotator.run()


if __name__ == "__main__":
    main()
