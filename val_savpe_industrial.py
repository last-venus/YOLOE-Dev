import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/jingxiuya/YOLOE-Dev")

from industrial_savpe import MaskAwareVisualPrompt, replace_load_visual_prompt
from ultralytics import YOLOE
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.yoloe.val import YOLOESegValidator


class IndustrialYOLOESegValidator(YOLOESegValidator):
    """工业场景 visual-val 验证器。

    与默认 YOLOE validator 的差别：
    1. visual prompt embedding 从参考集提取
    2. prompt 生成时优先使用实例 mask
    """

    def get_vpe_dataloader(self, data):
        # refer_split 用来指定“从哪一部分数据里提取 visual prompt embedding”
        # 常见选择：
        # - train：最常见，表示用训练集样本做 reference
        # - val：调试时可用，但更容易高估效果
        refer_split = getattr(self.args, "refer_split", self.args.split)
        dataset = build_yolo_dataset(
            self.args,
            data.get(refer_split, data.get("val")),
            self.args.batch,
            data,
            mode="val",
            rect=False,
        )
        replace_load_visual_prompt(dataset, prompt_cls=MaskAwareVisualPrompt)

        return build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )


DEFAULT_DATASET_YAML = "/home/jingxiuya/datasets/african-wildlife-seg-overfit12/african-wildlife-seg-overfit12.yaml"
DEFAULT_RUN_DIR = Path("/home/jingxiuya/YOLOE-Dev/runs/savpe/african-wildlife-industrial-vp-overfit12")
DEFAULT_WEIGHTS = DEFAULT_RUN_DIR / "weights" / "best.pt"


def _as_overrides_dict(base_overrides=None):
    """把 dict / argparse namespace / trainer args 统一转成 dict。"""
    if base_overrides is None:
        return {}
    if isinstance(base_overrides, dict):
        return dict(base_overrides)
    return dict(vars(base_overrides))


def build_visual_val_overrides(
    *,
    data,
    split="val",
    batch=4,
    imgsz=640,
    device="0",
    workers=0,
    project=None,
    name=None,
    plots=False,
    base_overrides=None,
):
    """构造可复用的 visual-val 参数。

    这些参数既可以给单独验证脚本使用，也可以给训练器里的自动 visual-val 使用。
    """
    overrides = _as_overrides_dict(base_overrides)
    overrides.update(
        {
            "mode": "val",
            "data": data,
            "split": split,
            "batch": batch,
            "imgsz": imgsz,
            "workers": workers,
            "device": device,
            "plots": plots,
            "rect": False,
        }
    )
    if project is not None:
        overrides["project"] = project
    if name is not None:
        overrides["name"] = name
    return overrides


def create_visual_val_validator(
    *,
    data,
    split="val",
    batch=4,
    imgsz=640,
    device="0",
    workers=0,
    project=None,
    name=None,
    plots=False,
    refer_split="train",
    base_overrides=None,
    callbacks=None,
):
    """创建工业场景 visual-val validator。"""
    validator_args = build_visual_val_overrides(
        data=data,
        split=split,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=project,
        name=name,
        plots=plots,
        base_overrides=base_overrides,
    )
    validator = IndustrialYOLOESegValidator(args=validator_args, _callbacks=callbacks)
    validator.args.refer_split = refer_split
    return validator


def run_visual_val(
    model_or_weights,
    *,
    data,
    refer_data=None,
    refer_split="train",
    split="val",
    batch=4,
    imgsz=640,
    device="0",
    workers=0,
    project=None,
    name=None,
    plots=False,
    base_overrides=None,
    callbacks=None,
):
    """运行可复用的工业场景 visual-val。

    常改参数：
    - data: 评估所用数据 yaml
    - refer_data: 提取 visual prompt embedding 的参考数据 yaml
    - refer_split: 从参考数据的哪个 split 提取 embedding，默认 train
    - split: 真正评估哪个 split，默认 val
    - batch / imgsz / device / workers: 常规验证参数

    返回：
    1. `stats`：validator 直接返回的统计字典
    2. `validator`：可继续读取 `metrics.results_dict`、`save_dir` 等信息
    """
    model_ref = model_or_weights
    if isinstance(model_or_weights, YOLOE):
        model_ref = model_or_weights.model
        callbacks = callbacks or model_or_weights.callbacks
        base_overrides = base_overrides or model_or_weights.overrides

    validator = create_visual_val_validator(
        data=data,
        split=split,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=project,
        name=name,
        plots=plots,
        refer_split=refer_split,
        base_overrides=base_overrides,
        callbacks=callbacks,
    )
    stats = validator(model=model_ref, load_vp=True, refer_data=refer_data or data)
    return stats, validator


def parse_args():
    parser = argparse.ArgumentParser(description="Industrial SAVPE visual-prompt validation.")
    parser.add_argument("--data", default=DEFAULT_DATASET_YAML, help="数据集 yaml 路径。")
    parser.add_argument(
        "--refer-data",
        default=None,
        help="提取 visual prompt embedding 的参考数据 yaml。默认与 --data 相同。",
    )
    parser.add_argument(
        "--refer-split",
        default="train",
        choices=["train", "val", "test"],
        help="提取 visual prompt embedding 所使用的数据划分。通常保持 train。",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="待验证权重路径。")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="评估所使用的数据划分。")
    parser.add_argument("--batch", type=int, default=4, help="验证 batch size。")
    parser.add_argument("--imgsz", type=int, default=640, help="验证分辨率。")
    parser.add_argument("--device", default="0", help="验证设备。")
    parser.add_argument("--workers", type=int, default=0, help="dataloader worker 数。")
    parser.add_argument("--project", default=str(DEFAULT_RUN_DIR.parent), help="验证结果输出目录。")
    parser.add_argument("--name", default=f"{DEFAULT_RUN_DIR.name}-vp-val", help="验证实验名。")
    parser.add_argument("--plots", action="store_true", help="是否保存可视化图。")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLOE(args.weights)
    stats, validator = run_visual_val(
        model,
        data=args.data,
        refer_data=args.refer_data,
        refer_split=args.refer_split,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        plots=args.plots,
    )
    metrics = validator.metrics

    print("save_dir:", metrics.save_dir)
    print("stats:", stats)
    print("results_dict:", metrics.results_dict)


if __name__ == "__main__":
    main()
