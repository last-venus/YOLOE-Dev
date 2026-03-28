import sys
sys.path.insert(0, "/home/jingxiuya/YOLOE-Dev")

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

# Initialize a detection model from a config
model = YOLOE("yoloe-26s.yaml")

# Load weights from a pretrained segmentation checkpoint (same scale)
model.load("/home/jingxiuya/datasets/pt/yoloe-26s-seg.pt")

# Fine-tune on your detection dataset
results = model.train(
    data="/home/jingxiuya/datasets/african-wildlife/african-wildlife.yaml",  # Segmentation dataset
    epochs=40,
    patience=10,
    trainer=YOLOEPETrainer,  # <- Important: use detection trainer
    optimizer='AdamW', 
    lr0=1e-3,

    batch=8,
    # nbs=4,          # 关键：关掉大梯度累积
    workers=0,
    device=0,

    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,
    copy_paste=0.0,
    scale=0.0,
    translate=0.0,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    fliplr=0.0,
    flipud=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
)
