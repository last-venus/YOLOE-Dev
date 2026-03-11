from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# 直接填写示例图片路径
IMAGE = "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_grid.jpg"  # 替换为你的输入图像路径

# 待推理图片所在文件夹
INFER_DIR = "/home/jingxiuya/YOLOE/scripts/images/20260312/data"

# 推理结果保存文件夹
OUTPUT_DIR = "/home/jingxiuya/YOLOE/scripts/runs/260312/0000_26m_multi_vpe_flower"

# 模型权重路径
MODEL_PATH = "/home/jingxiuya/YOLOE/pt/yoloe-26m-seg.pt"

# 推理设备: 有 GPU 可填 0, 无 GPU 可改成 "cpu"
DEVICE = 0

# 视觉提示推理通常需要较低阈值, 否则容易直接被过滤掉
CONF = 0.1

# 支持的图片格式
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 定义视觉提示框, 坐标基于示例图片 IMAGE
# visual_prompts = dict(
#     bboxes=np.array(
#         [
#             [529.5, 337.6, 657.5, 505.6],  # paper cup
#             [299.1, 318.3, 490.6, 525.9],  # red cup
#             # [728.9, 129.3, 862.3, 507.2], # black cup
#             # [1279.0, 891.4, 2664.6, 2255.3],  # red cup only
#             # [1878.9, 1105.8, 2775.6, 2229.9],  # paper cup only
#         ]
#     ),
#     cls=np.array([0, 1]),
# )
# visual_prompts = dict(
#     bboxes=np.array(
#         [
#             [594.3, 286.5, 827.7, 576.0],     # Box 18, class 0
#             [1903.9, 222.3, 2211.1, 604.5],   # Box 7, class 0
#             [3158.1, 74.8, 3407.2, 392.1],    # Box 10, class 0
#             [594.6, 970.4, 844.6, 1293.8],    # Box 3, class 0
#             [1823.3, 953.7, 2092.9, 1305.4],  # Box 5, class 0
#             [2908.6, 1028.2, 3135.9, 1316.5], # Box 16, class 0
#             [530.2, 1721.4, 822.8, 2077.2],   # Box 4, class 0
#             [1969.3, 1639.5, 2265.7, 2003.1], # Box 2, class 0
#             [3310.8, 1552.5, 3597.2, 1908.6], # Box 12, class 0

#             [107.9, 1630.0, 468.0, 2045.8],   # Box 1, class 1
#             [1488.4, 1619.3, 1946.6, 2091.4], # Box 6, class 1
#             [283.7, 863.5, 544.9, 1215.4],    # Box 8, class 1
#             [2851.3, 1527.9, 3344.5, 2020.3], # Box 9, class 1
#             [1459.8, 145.0, 1816.1, 580.9],   # Box 11, class 1
#             [2831.2, 0.8, 3099.4, 346.9],     # Box 13, class 1
#             [239.5, 250.5, 544.1, 596.5],     # Box 14, class 1
#             [1526.9, 816.3, 1796.9, 1184.5],  # Box 15, class 1
#             [2682.8, 909.8, 2933.6, 1223.4],  # Box 17, class 1
#         ]
#     ),
#     cls=np.array([
#         0, 0, 0, 0, 0, 0, 0, 0, 0,
#         1, 1, 1, 1, 1, 1, 1, 1, 1
#     ]),
# )

# visual_prompts = dict(
#     bboxes=np.array(
#         [
#             [729.6, 177.6, 1068.0, 463.2],
#             [748.8, 496.8, 984.0, 734.4],
#             [1632.0, 374.4, 1862.4, 573.6],
#             [1891.2, 316.8, 2126.4, 525.6],
#             [2812.8, 247.2, 3055.2, 477.6],
#             [3028.8, 420.0, 3285.6, 650.4],
#             [650.4, 940.8, 892.8, 1135.2],
#             [784.8, 1118.4, 996.0, 1293.6],
#             [2052.0, 916.8, 2313.6, 1156.8],
#             [2028.0, 1156.8, 2282.4, 1346.4],
#             [2640.0, 830.4, 3019.2, 1156.8],
#             [2695.2, 1012.8, 3074.4, 1358.4],
#             [237.6, 1711.2, 472.8, 1936.8],
#             [386.4, 1704.0, 652.8, 1953.6],
#             [1740.0, 1687.2, 1999.2, 1927.2],
#             [2239.2, 1680.0, 2467.2, 1881.6],
#             [2690.4, 1848.0, 2923.2, 1999.2],
#             [2959.2, 1898.4, 3170.4, 2054.4],
#             [420.0, 211.2, 715.2, 578.4],
#             [1708.8, 127.2, 2004.0, 324.0],
#             [3055.2, 156.0, 3360.0, 427.2],
#             [2908.8, 758.4, 3355.2, 1228.8],
#             [1773.6, 883.2, 2100.0, 1204.8],
#             [362.4, 1041.6, 760.8, 1365.6],
#             [216.0, 1519.2, 607.2, 1764.0],
#             [1936.8, 1636.8, 2289.6, 2008.8],
#             [2892.0, 1716.0, 3170.4, 1886.4],
#         ],
#         dtype=np.float32,
#     ),
#     cls=np.array(
#         [
#             0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0,
#             1, 1, 1, 1, 1, 1, 1, 1, 1,
#         ],
#         dtype=np.int32,
#     ),
# )
visual_prompts = dict(
    bboxes=np.array(
        [
            [16.7, 8.3, 508.3, 481.7],
            [570.0, 10.0, 1078.3, 463.3],
            [1135.0, 38.3, 1461.7, 413.3],
            [100.0, 603.3, 525.0, 926.7],
            [615.0, 525.0, 1028.3, 955.0],
            [1295.0, 578.3, 1570.0, 876.7],
            [131.7, 1011.7, 411.7, 1326.7],
            [591.7, 1013.3, 993.3, 1463.3],
        ],
        dtype=np.float32,
    ),
    cls=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
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

    for box, cls_id in zip(prompts["bboxes"], prompts["cls"]):
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
            f"CLS {int(cls_id)}",
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
