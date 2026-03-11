from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# ============================================================
# 这个脚本用于演示:
# 1. 输入多张参考图, 每张图单独配置 visual prompt 框
# 2. 每张参考图单独提取一次 VPE(Visual Prompt Embedding)
# 3. 按类别聚合多个参考图得到的 VPE
# 4. 用聚合后的 VPE 在测试集文件夹上做批量推理
# 5. 保存参考图标注结果和测试集推理结果
#
# 使用方式:
# 1. 先按你的数据修改下面的配置区
# 2. 运行: python scripts/vision_prompt_multi_vpe.py
# ============================================================


# ----------------------------
# 基础配置
# ----------------------------

# 模型权重路径
MODEL_PATH = "/home/jingxiuya/YOLOE/pt/yoloe-26s-seg.pt"

# 待推理测试集文件夹
INFER_DIR = "/home/jingxiuya/YOLOE/scripts/images/20260312/data"

# 输出目录
OUTPUT_DIR = "/home/jingxiuya/YOLOE/scripts/runs/260312/0001_26s_multi_vpe_flower"

# 推理设备: 有 GPU 可填 0, 无 GPU 可改成 "cpu"
DEVICE = 0

# 视觉提示推理通常建议阈值不要太高
CONF = 0.1

# 支持的图片后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------
# 类别定义
# ----------------------------
# 这里的顺序必须和 cls 编号一致:
# cls=0 对应 CLASS_NAMES[0]
# cls=1 对应 CLASS_NAMES[1]
CLASS_NAMES = [
    "flower0",
    "flower1",
]


# ----------------------------
# 多张参考图配置
# ----------------------------
# 每个元素代表一张参考图:
# - image: 参考图路径
# - prompts.bboxes: 该图上的提示框
# - prompts.cls: 每个提示框对应的类别编号
#
# 下面先给一个四张图的写法示例。你可以继续往里加, 也可以删减。
# 如果某张参考图里有多个框, 只要 bboxes 和 cls 的长度一致即可。
REFERENCE_SPECS = [
    {
        "image": "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_01451.jpg",
        "prompts": dict(
            bboxes=np.array(
                [
                    [14.0, 19.0, 504.0, 472.0],
                ],
                dtype=np.float32,
            ),
            cls=np.array([0], dtype=np.int32),
        ),
    },
    {
        "image": "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_01456.jpg",
        "prompts": dict(
            bboxes=np.array(
                [
                    [22.0, 31.0, 467.0, 450.0],
                ],
                dtype=np.float32,
            ),
            cls=np.array([0], dtype=np.int32),
        ),
    },
    {
        "image": "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_01463.jpg",
        "prompts": dict(
            bboxes=np.array(
                [
                    [65.0, 56.0, 485.0, 410.0],
                ],
                dtype=np.float32,
            ),
            cls=np.array([0], dtype=np.int32),
        ),
    },
    {
        "image": "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_01464.jpg",
        "prompts": dict(
            bboxes=np.array(
                [
                    [90.0, 148.0, 477.0, 525.0],
                ],
                dtype=np.float32,
            ),
            cls=np.array([0], dtype=np.int32),
        ),
    },
    {
        "image": "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_02135.jpg",
        "prompts": dict(
            bboxes=np.array(
                [
                    [95.0, 22.0, 540.0, 449.0],
                ],
                dtype=np.float32,
            ),
            cls=np.array([1], dtype=np.int32),
        ),
    },
    {
        "image": "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_02136.jpg",
        "prompts": dict(
            bboxes=np.array(
                [
                    [221.0, 94.0, 489.0, 384.0],
                ],
                dtype=np.float32,
            ),
            cls=np.array([1], dtype=np.int32),
        ),
    },
    {
        "image": "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_02137.jpg",
        "prompts": dict(
            bboxes=np.array(
                [
                    [161.0, 4.0, 491.0, 320.0],
                ],
                dtype=np.float32,
            ),
            cls=np.array([1], dtype=np.int32),
        ),
    },
    {
        "image": "/home/jingxiuya/YOLOE/scripts/images/20260312/prompt/image_02140.jpg",
        "prompts": dict(
            bboxes=np.array(
                [
                    [56.0, 34.0, 518.0, 451.0],
                ],
                dtype=np.float32,
            ),
            cls=np.array([1], dtype=np.int32),
        ),
    },
]
# REFERENCE_SPECS = [
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0002.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [725.0, 154.0, 1054.0, 475.0],  # small
#                     [749.0, 496.0, 988.0, 732.0],   # small
#                     [410.0, 214.0, 715.0, 569.0],   # big
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0005.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [604.0, 307.0, 840.0, 527.0],  # small
#                     [353.0, 367.0, 585.0, 592.0],  # small
#                     [428.0, 115.0, 714.0, 329.0],  # big
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0007.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [259.0, 253.0, 495.0, 486.0],  # small
#                     [476.0, 418.0, 709.0, 645.0],  # small
#                     [487.0, 158.0, 793.0, 433.0],  # big
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0009.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [658.0, 216.0, 886.0, 410.0],  # small
#                     [799.0, 387.0, 988.0, 574.0],  # small
#                     [375.0, 322.0, 749.0, 636.0],  # big
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0013.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [771.0, 201.0, 1034.0, 436.0],  # small
#                     [741.0, 428.0, 999.0, 631.0],   # small
#                     [498.0, 156.0, 815.0, 492.0],   # big
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0019.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [79.0, 110.0, 446.0, 449.0],   # small
#                     [149.0, 281.0, 518.0, 640.0],  # small
#                     [339.0, 36.0, 789.0, 528.0],   # big
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0022.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [243.0, 269.0, 473.0, 497.0],  # paper_cup
#                     [374.0, 248.0, 652.0, 517.0],  # paper_cup
#                     [220.0, 89.0, 598.0, 324.0],   # red_cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0030.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [459.0, 250.0, 735.0, 491.0],   # paper_cup
#                     [967.0, 229.0, 1173.0, 449.0],  # paper_cup
#                     [658.0, 205.0, 1000.0, 558.0],  # red_cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20260311/prompt/0033.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [133.0, 409.0, 354.0, 557.0],  # paper_cup
#                     [405.0, 457.0, 602.0, 604.0],  # paper_cup
#                     [336.0, 287.0, 610.0, 444.0],  # red_cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 0, 1], dtype=np.int32),
#         ),
#     },


# ]
# REFERENCE_SPECS = [
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0000.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [594.9, 287.1, 826.9, 577.5],  # paper cup
#                     [238.6, 250.5, 542.1, 597.6],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0001.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [620.5, 221.6, 931.7, 606.4],  # paper cup
#                     [178.1, 144.9, 532.8, 582.6],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0002.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [601.1, 75.8, 846.5, 394.3],  # paper cup
#                     [272.7, 3.5, 539.9, 347.3],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
#     {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0003.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [596.6, 250.5, 847.6, 576.7],  # paper cup
#                     [282.7, 146.1, 545.4, 496.7],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
#      {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0004.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [541.8, 235.2, 815.0, 589.5],  # paper cup
#                     [244.4, 95.8, 519.8, 466.4],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
#      {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0005.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [348.6, 308.3, 572.1, 597.9],  # paper cup
#                     [124.1, 191.1, 376.8, 507.1],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
#      {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0006.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [528.6, 283.2, 823.8, 638.6],  # paper cup
#                     [106.9, 189.2, 469.6, 609.3],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
#          {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0007.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [688.8, 200.3, 985.9, 564.3],  # paper cup
#                     [208.4, 181.1, 666.6, 652.2],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
#          {
#         "image": "/home/jingxiuya/YOLOE/scripts/images/20250310/data/0008.jpg",
#         "prompts": dict(
#             bboxes=np.array(
#                 [
#                     [752.4, 113.0, 1038.0, 469.0],  # paper cup
#                     [292.5, 90.6, 784.7, 580.8],  # red cup
#                 ],
#                 dtype=np.float32,
#             ),
#             cls=np.array([0, 1], dtype=np.int32),
#         ),
#     },
# ]


def collect_images(image_dir: Path) -> list[Path]:
    """收集测试集目录中的所有图片。"""
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def clone_prompts(prompts: dict) -> dict:
    """复制一份 prompts，避免 predictor 在内部 pop 掉原字典里的字段。"""
    cloned = {
        "cls": np.array(prompts["cls"], dtype=np.int32, copy=True),
    }
    if "bboxes" in prompts:
        cloned["bboxes"] = np.array(prompts["bboxes"], dtype=np.float32, copy=True)
    if "masks" in prompts:
        cloned["masks"] = np.array(prompts["masks"], copy=True)
    return cloned


def validate_reference_specs(reference_specs: list[dict], class_names: list[str]) -> None:
    """在正式跑推理前检查配置, 提前把明显问题报出来。"""
    if not reference_specs:
        raise ValueError("REFERENCE_SPECS 不能为空, 至少要配置一张参考图。")

    for index, spec in enumerate(reference_specs, start=1):
        image_path = Path(spec["image"])
        prompts = spec["prompts"]
        bboxes = prompts.get("bboxes")
        cls_ids = prompts.get("cls")

        if not image_path.exists():
            raise FileNotFoundError(f"第 {index} 张参考图不存在: {image_path}")
        if bboxes is None or len(bboxes) == 0:
            raise ValueError(f"第 {index} 张参考图没有配置 bboxes: {image_path}")
        if cls_ids is None or len(cls_ids) == 0:
            raise ValueError(f"第 {index} 张参考图没有配置 cls: {image_path}")
        if len(bboxes) != len(cls_ids):
            raise ValueError(
                f"第 {index} 张参考图的 bboxes 数量和 cls 数量不一致: "
                f"{len(bboxes)} vs {len(cls_ids)}"
            )

        for cls_id in cls_ids:
            if int(cls_id) < 0 or int(cls_id) >= len(class_names):
                raise ValueError(
                    f"第 {index} 张参考图里存在越界 cls={int(cls_id)}, "
                    f"当前 CLASS_NAMES 只有 {len(class_names)} 个类别。"
                )


def draw_prompt_boxes(
    image_path: Path,
    output_path: Path,
    prompts: dict,
    class_names: list[str],
) -> None:
    """把每张参考图上的 prompt 框画出来, 便于复核输入是否正确。"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load prompt image: {image_path}")
        return

    for box, cls_id in zip(prompts["bboxes"], prompts["cls"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_id = int(cls_id)
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        label = f"CLS {cls_id}: {cls_name}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

        corner_len = 24
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 3)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 3)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 3)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 3)

        cv2.putText(
            img,
            label,
            (x1, max(40, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), img)
    print(f"参考图可视化已保存: {output_path}")


def initialize_predictor(model: YOLOE, first_image: Path, first_prompts: dict) -> None:
    """先跑一次最小预测, 确保 model.predictor 被正确创建出来。"""
    model.predict(
        source=str(first_image),
        visual_prompts=clone_prompts(first_prompts),
        predictor=YOLOEVPSegPredictor,
        device=DEVICE,
        conf=CONF,
        save=False,
        verbose=False,
    )


def extract_reference_vpe(model: YOLOE, image_path: Path, prompts: dict) -> torch.Tensor:
    """从单张参考图中提取 VPE。

    返回张量形状通常为:
    (1, N, D)
    其中:
    - 1 表示单张图片
    - N 表示这张图里 prompt 的数量
    - D 表示 embedding 维度
    """
    model.predictor.set_prompts(clone_prompts(prompts))
    return model.predictor.get_vpe(str(image_path))


def aggregate_vpes_by_class(
    per_reference_vpes: list[torch.Tensor],
    reference_specs: list[dict],
    class_names: list[str],
) -> tuple[torch.Tensor, dict[int, int]]:
    """按类别聚合多张参考图的 VPE。

    设计思路:
    - 每张参考图可以有多个 prompt
    - 但 YOLOE 在构造 visual prompt 时，会先按 cls 做 unique
    - 同一张图里，同一类的多个框会先合并成一个 visual，再提取一个 VPE
    - 所以 get_vpe() 返回的不是“框数”，而是“该图里的唯一类别数”
    - 最后我们把不同参考图里、同一类得到的 VPE 再做均值聚合
    """
    cls_to_embeddings = defaultdict(list)
    cls_to_prompt_count = defaultdict(int)

    for spec, raw_vpe in zip(reference_specs, per_reference_vpes):
        cls_ids = np.array(spec["prompts"]["cls"], dtype=np.int32)
        unique_cls_ids = np.unique(cls_ids)

        # get_vpe 返回形状一般为 (1, N, D)，这里去掉 batch 维
        # 这里的 N 不是框数，而是该图中 unique(cls) 的数量。
        raw_vpe = raw_vpe.squeeze(0)

        if raw_vpe.ndim != 2:
            raise ValueError(f"提取到的 VPE 维度异常: {tuple(raw_vpe.shape)}")
        if raw_vpe.shape[0] != len(unique_cls_ids):
            raise ValueError(
                "VPE 数量和该图唯一类别数不一致: "
                f"vpe={raw_vpe.shape[0]}, unique_cls={len(unique_cls_ids)}, "
                f"cls_ids={cls_ids.tolist()}, unique={unique_cls_ids.tolist()}"
            )

        for vpe_index, cls_id in enumerate(unique_cls_ids):
            cls_id = int(cls_id)
            cls_to_embeddings[cls_id].append(raw_vpe[vpe_index])
            cls_to_prompt_count[cls_id] += int((cls_ids == cls_id).sum())

    final_embeddings = []
    for cls_id, cls_name in enumerate(class_names):
        if not cls_to_embeddings[cls_id]:
            raise ValueError(
                f"类别 {cls_id} ({cls_name}) 没有任何参考 prompt, 无法聚合出最终 VPE。"
            )

        stacked = torch.stack(cls_to_embeddings[cls_id], dim=0)
        mean_embedding = stacked.mean(dim=0, keepdim=True)
        mean_embedding = F.normalize(mean_embedding, p=2, dim=-1)
        final_embeddings.append(mean_embedding)

    final_vpe = torch.stack(final_embeddings, dim=0).transpose(0, 1)
    return final_vpe, dict(cls_to_prompt_count)


def save_vpe_summary(
    output_dir: Path,
    class_names: list[str],
    cls_to_prompt_count: dict[int, int],
    final_vpe: torch.Tensor,
) -> None:
    """保存一份简单的文本说明, 方便回看这次聚合使用了多少样本。"""
    summary_path = output_dir / "vpe_summary.txt"
    lines = [
        "Multi-reference VPE aggregation summary",
        f"final_vpe.shape = {tuple(final_vpe.shape)}",
        "",
    ]
    for cls_id, cls_name in enumerate(class_names):
        lines.append(
            f"cls={cls_id}, name={cls_name}, prompt_count={cls_to_prompt_count.get(cls_id, 0)}"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"VPE 聚合信息已保存: {summary_path}")


def run_inference_on_dataset(
    model: YOLOE,
    image_files: list[Path],
    output_dir: Path,
) -> None:
    """使用聚合后的 VPE 对测试集逐张推理并保存结果。"""
    for image_file in image_files:
        print(f"正在推理测试图: {image_file}")
        model.predict(
            source=str(image_file),
            save=True,
            device=DEVICE,
            conf=CONF,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
        )


def set_classes_with_embeddings(
    model: YOLOE,
    class_names: list[str],
    embeddings: torch.Tensor,
) -> None:
    """兼容不同 ultralytics 版本的 set_classes 行为。

    某些版本的 YOLOE.set_classes() 默认假设 model.names 是 dict，
    但实际运行时 model.names 可能是 list，这时调用包装层会报:
    AttributeError: 'list' object has no attribute 'values'

    所以这里直接走底层 model.model.set_classes()，并手动同步 predictor 的 names。
    """
    raw_names = getattr(model.model, "names", None)
    if isinstance(raw_names, dict):
        current_names = list(raw_names.values())
    elif isinstance(raw_names, list):
        current_names = list(raw_names)
    else:
        current_names = []

    if sorted(current_names) != sorted(class_names):
        model.model.set_classes(class_names, embeddings)

    if model.predictor:
        model.predictor.model.names = model.model.names


def main() -> None:
    infer_dir = Path(INFER_DIR)
    output_dir = Path(OUTPUT_DIR)
    prompt_vis_dir = output_dir / "prompt_refs"

    validate_reference_specs(REFERENCE_SPECS, CLASS_NAMES)

    if not infer_dir.exists() or not infer_dir.is_dir():
        raise NotADirectoryError(f"测试集目录不存在: {infer_dir}")

    image_files = collect_images(infer_dir)
    if not image_files:
        raise FileNotFoundError(f"测试集目录里没有图片: {infer_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    model = YOLOE(MODEL_PATH)

    # 2. 先初始化 predictor
    first_spec = REFERENCE_SPECS[0]
    initialize_predictor(model, Path(first_spec["image"]), first_spec["prompts"])

    # 3. 对每张参考图单独提取 VPE，同时把参考图输入也可视化保存下来
    per_reference_vpes = []
    print("开始逐张参考图提取 VPE...")
    for ref_index, spec in enumerate(REFERENCE_SPECS, start=1):
        ref_image_path = Path(spec["image"])
        prompts = spec["prompts"]

        print(f"[参考图 {ref_index}] 提取 VPE: {ref_image_path}")
        raw_vpe = extract_reference_vpe(model, ref_image_path, prompts)
        per_reference_vpes.append(raw_vpe)

        prompt_vis_path = prompt_vis_dir / f"{ref_index:02d}_{ref_image_path.name}"
        draw_prompt_boxes(
            image_path=ref_image_path,
            output_path=prompt_vis_path,
            prompts=clone_prompts(prompts),
            class_names=CLASS_NAMES,
        )

    # 4. 把多张参考图提取到的 VPE 按类别做聚合
    final_vpe, cls_to_prompt_count = aggregate_vpes_by_class(
        per_reference_vpes=per_reference_vpes,
        reference_specs=REFERENCE_SPECS,
        class_names=CLASS_NAMES,
    )

    # 保存聚合后的 embedding，方便后续直接复用或分析
    vpe_save_path = prompt_vis_dir / "aggregated_vpe.pt"
    torch.save(final_vpe.detach().cpu(), vpe_save_path)
    print(f"聚合后的 VPE 已保存: {vpe_save_path}")

    save_vpe_summary(prompt_vis_dir, CLASS_NAMES, cls_to_prompt_count, final_vpe)

    # 5. 将聚合后的 VPE 注入模型，后续推理就会使用它作为类别表示
    # 这里显式覆盖 is_fused，是为了和你给的参考代码保持一致，避免某些离线推理场景报 fused 相关错误。
    model.is_fused = lambda: False
    set_classes_with_embeddings(model, CLASS_NAMES, final_vpe)
    print("已将聚合后的 VPE 设置到模型中。")

    # 6. 在测试集上批量推理并保存结果
    run_inference_on_dataset(model, image_files, output_dir)

    print(f"全部完成。推理结果目录: {output_dir}")
    print(f"参考图可视化目录: {prompt_vis_dir}")


if __name__ == "__main__":
    main()
