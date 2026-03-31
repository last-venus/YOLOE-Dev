import sys

sys.path.insert(0, "/home/jingxiuya/YOLOE-Dev")

from industrial_savpe import IndustrialSAVPETrainer
from ultralytics import YOLOE
from ultralytics.utils import LOGGER, LOCAL_RANK, RANK
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first
from val_savpe_industrial import run_visual_val


DATASET_YAML = "/home/jingxiuya/datasets/african-wildlife-seg-overfit12/african-wildlife-seg-overfit12.yaml"
WEIGHTS = "/home/jingxiuya/datasets/pt/yoloe-26s-seg.pt"


def resolve_single_yolo_data(data_cfg, split):
    """从 world 风格 data 配置中取出单个 yolo_data yaml。"""
    split_cfg = data_cfg.get(split, {})
    yolo_data = split_cfg.get("yolo_data", [])
    if len(yolo_data) != 1:
        raise ValueError(f"Expected exactly one yolo_data entry for split '{split}', but got {len(yolo_data)}.")
    return yolo_data[0]


class IndustrialVisualValTrainer(IndustrialSAVPETrainer):
    """把默认 text-val 换成 visual-val 的工业场景 trainer。"""

    # 如果后面换参考集或换验证 split，通常改这几个类属性就够了。
    visual_val_refer_data = DATASET_YAML
    visual_val_refer_split = "train"
    visual_val_split = "val"
    visual_val_batch = 4
    visual_val_name = "african-wildlife-industrial-vp-overfit12-visual-val"

    def _build_visual_val_kwargs(self):
        eval_data = resolve_single_yolo_data(self.args.data, "val")
        refer_data = self.visual_val_refer_data or resolve_single_yolo_data(self.args.data, "train")
        return {
            "data": eval_data,
            "refer_data": refer_data,
            "refer_split": self.visual_val_refer_split,
            "split": self.visual_val_split,
            "batch": self.visual_val_batch,
            "imgsz": self.args.imgsz,
            "device": self.args.device,
            "workers": self.args.workers,
            "project": self.save_dir.parent,
            "name": self.visual_val_name,
            "plots": self.args.plots,
            "base_overrides": self.args,
            "callbacks": self.callbacks,
        }

    def validate(self):
        """训练中如果开启 val，则跑 visual-val。"""
        stats, _ = run_visual_val(self.ema.ema, **self._build_visual_val_kwargs())
        fitness = stats.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return stats, fitness

    def final_eval(self):
        """训练结束后对 best.pt 再跑一次最终 visual-val。"""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})

        if model:
            LOGGER.info(f"\nVisual-validating {model}...")
            stats, _ = run_visual_val(model, **self._build_visual_val_kwargs())
            self.metrics = stats
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")


DATA = {
    "train": {"yolo_data": [DATASET_YAML]},
    "val": {"yolo_data": [DATASET_YAML]},
}


model = YOLOE(WEIGHTS)
freeze = []


results = model.train(
    cls=1.0,  # 分类损失权重；如不训练分类可改为 0。
    data=DATA,  # world 风格数据配置，兼容 YOLOE 的 VP trainer。
    trainer=IndustrialVisualValTrainer,  # 用工业版 trainer，并把默认验证改成 visual-val。
    batch=4,  # overfit12 时小 batch 往往更稳，也更省显存。
    nbs=4,  # 名义 batch size 与真实 batch 对齐，避免默认梯度累积。
    epochs=200,  # 过拟合测试时先给足 epoch，重点看能否真正记住样本。
    patience=10,  # 不提前停止，方便直接看 overfit 能力。
    imgsz=640,  # 输入分辨率；通常和预训练模型保持一致。
    optimizer="AdamW",  # 如需试 SGD/Adam，可改这里。
    lr0=3e-4,  # overfit12 时可适当调大，帮助更快贴合小样本。
    lrf=1.0,  # 近似常数学习率，适合这种小规模 overfit 测试。
    warmup_epochs=0.0,  # 关闭 warmup，让模型尽快进入有效优化。
    warmup_bias_lr=0.0,  # 与上面保持一致。
    weight_decay=0.0,  # 过拟合测试一般先关掉正则，便于观察模型上限。
    momentum=0.9,  # 动量参数；如切换优化器可一并调整。
    workers=0,  # 调试/小数据训练时更稳。
    device=0,  # 单卡训练；如需改 CPU 或其他卡，改这里。
    freeze=freeze,  # overfit12 里先全部放开，更容易验证链路上限。
    val=True,  # 这里会走上面自定义的 visual-val，而不是默认 text-val。
    amp=False,  # 过拟合测试优先稳定性，先关闭 AMP。
    mosaic=0.0,  # 过拟合测试通常关闭强增强。
    mixup=0.0,  # 同上。
    copy_paste=0.0,  # 同上。
    scale=0.0,  # 同上。
    translate=0.0,  # 同上。
    degrees=0.0,  # 同上。
    shear=0.0,  # 同上。
    perspective=0.0,  # 同上。
    fliplr=0.0,  # 同上；如果你想保留最轻量增强，可以只打开这一项。
    flipud=0.0,  # 同上。
    hsv_h=0.0,  # 同上。
    hsv_s=0.0,  # 同上。
    hsv_v=0.0,  # 同上。
    erasing=0.0,  # 同上。
    project="/home/jingxiuya/YOLOE-Dev/runs/savpe",  # 输出目录；重复实验时常改。
    name="african-wildlife-industrial-vp-overfit12",  # 实验名；重复实验时常改。
)
