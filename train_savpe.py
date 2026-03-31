import sys

sys.path.insert(0, "/home/jingxiuya/YOLOE-Dev")

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOESegVPTrainer


DATASET_YAML = "/home/jingxiuya/datasets/african-wildlife-seg-overfit12/african-wildlife-seg-overfit12.yaml"
WEIGHTS = "/home/jingxiuya/datasets/pt/yoloe-26l-seg.pt"

# YOLOESegVPTrainer 走的是 YOLO-World 风格的数据配置，
# 这里必须传带有 train/val 结构的字典，不能直接传单个 yaml 路径字符串。
DATA = {
    "train": {"yolo_data": [DATASET_YAML]},
    "val": {"yolo_data": [DATASET_YAML]},
}

# 加载预训练分割权重。
model = YOLOE(WEIGHTS)

head_index = len(model.model.model) - 1
# 进入更激进的 overfit 模式：
# 除了 DFL 这个本身基本不需要训练的模块，其余 backbone、neck、head 全部参与训练。
# 这样可以最大限度保留 SAVPE，同时验证这条训练路径能否记住小数据集。
freeze = []
for name, child in model.model.model[-1].named_children():
    # DFL 的积分卷积通常保持冻结即可，其余 head 模块全部打开。
    if name in {"dfl"}:
        freeze.append(f"{head_index}.{name}")


results = model.train(
    data=DATA,  # 训练数据配置，使用上面定义的 world 风格字典。
    batch=4,  # batch size 直接设成接近整个训练集大小，强化记忆而不是泛化。
    epochs=400,  # 过拟合检查时把轮数开大，给模型足够时间把小数据集记住。
    patience=0,  # 关闭 early stopping，避免还没过拟合就提前停掉。
    imgsz=640,  # 输入分辨率，和预训练模型保持一致。
    close_mosaic=0,  # 不在后期关闭 mosaic，因为这里从头到尾都不使用 mosaic。
    optimizer="AdamW",  # 优化器，沿用 YOLOE 官方脚本常用设置。
    lr0=5e-4,  # 过拟合模式下适当提高学习率，让全量参数更快贴合小数据集。
    lrf=1.0,  # 使用近似常数学习率，不做大幅衰减，方便持续记忆训练集。
    warmup_epochs=0.0,  # 关闭 warmup，避免前几轮更新过慢。
    warmup_bias_lr=0.0,  # bias 的 warmup 学习率也一并关闭。
    weight_decay=0.0,  # 过拟合检查时去掉权重衰减，避免正则化阻碍记忆。
    momentum=0.9,  # 动量参数。
    nbs=24,  # 名义 batch size 设成和真实 batch 一致，避免默认 64 带来过大的梯度累积。
    workers=0,  # dataloader 进程数，小数据调试时设成 0 更省心。
    trainer=YOLOESegVPTrainer,  # 使用分割版 SAVPE trainer。
    device=0,  # 使用第 0 张 GPU 做单卡过拟合测试。
    freeze=freeze,  # 只冻结 DFL，其余模块包括 SAVPE 全部参与训练。
    val=False,  # 先关闭训练期间默认验证，避免 text prompt 验证结果干扰判断。
    amp=False,  # 关闭混合精度，减少小数据过拟合时的数值不稳定因素。
    # 下面这些增强全部关闭，目的是让模型尽量记住这 12 张图本身。
    mosaic=0.0,  # 关闭 mosaic 拼图增强。
    mixup=0.0,  # 关闭 mixup。
    copy_paste=0.0,  # 关闭 copy-paste。
    scale=0.0,  # 关闭随机缩放。
    translate=0.0,  # 关闭随机平移。
    degrees=0.0,  # 关闭随机旋转。
    shear=0.0,  # 关闭随机错切。
    perspective=0.0,  # 关闭透视变换。
    fliplr=0.0,  # 关闭左右翻转。
    flipud=0.0,  # 关闭上下翻转。
    hsv_h=0.0,  # 关闭色调扰动。
    hsv_s=0.0,  # 关闭饱和度扰动。
    hsv_v=0.0,  # 关闭亮度扰动。
    erasing=0.0,  # 关闭随机擦除，避免破坏小数据集上的强记忆。
    project="/home/jingxiuya/YOLOE-Dev/runs/savpe",  # 训练输出目录。
    name="african-wildlife-overfit24-overfit-max",  # 新实验名，和前几版结果分开保存。
)
