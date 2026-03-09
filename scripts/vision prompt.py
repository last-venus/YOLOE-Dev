from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# 直接填写示例图片路径
IMAGE = "/home/jingxiuya/YOLOE/scripts/images/20260306/data/0000.jpg"  # 替换为你的输入图像路径

# 待推理图片所在文件夹
INFER_DIR = "/home/jingxiuya/YOLOE/scripts/images/20260306/data"

# 推理结果保存文件夹
OUTPUT_DIR = "/home/jingxiuya/YOLOE/scripts/runs/vision_batch"

# 模型权重路径
MODEL_PATH = "/home/jingxiuya/YOLOE/pt/yoloe-26n-seg.pt"

# 推理设备: 有 GPU 可填 0, 无 GPU 可改成 "cpu"
DEVICE = 0

# 视觉提示推理通常需要较低阈值, 否则容易直接被过滤掉
CONF = 0.001

# 支持的图片格式
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 定义视觉提示框, 坐标基于示例图片 IMAGE
visual_prompts = dict(
    bboxes=np.array(
        [
            [529.5, 337.6, 657.5, 505.6],
            [299.1, 318.3, 490.6, 525.9],
            [728.9, 129.3, 862.3, 507.2],
        ]
    ),
    cls=np.array([0,1,2]),
)


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def draw_prompt_boxes(image_path: Path, output_path: Path, prompts: dict) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load prompt image: {image_path}")
        return

    for i, box in enumerate(prompts["bboxes"], start=1):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

        corner_len = 32
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 4)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 4)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 4)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 4)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 4)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 4)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 4)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 4)
        cv2.putText(
            img,
            f"VISUAL PROMPT {i}",
            (x1, max(40, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), img)
    print(f"Prompt reference image saved to: {output_path}")


def main() -> None:
    image_path = Path(IMAGE)
    infer_dir = Path(INFER_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not image_path.exists():
        raise FileNotFoundError(f"Prompt image not found: {image_path}")
    if not infer_dir.exists() or not infer_dir.is_dir():
        raise NotADirectoryError(f"Infer directory not found: {infer_dir}")

    image_files = collect_images(infer_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in: {infer_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOE(MODEL_PATH)

    for image_file in image_files:
        print(f"Running inference on: {image_file}")
        model.predict(
            source=str(image_file),
            refer_image=str(image_path),
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            save=True,
            device=DEVICE,
            conf=CONF,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
        )

    prompt_save_path = output_dir / f"{image_path.stem}_prompt{image_path.suffix}"
    draw_prompt_boxes(image_path, prompt_save_path, visual_prompts)

    print(f"Finished. All result images are saved in: {output_dir}")


if __name__ == "__main__":
    main()
