import sys
sys.path.insert(0, "/home/jingxiuya/YOLOE-Dev")

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

model = YOLOE("/home/jingxiuya/datasets/pt/yoloe-26s-seg.pt")

results = model.train(
    data="/home/jingxiuya/datasets/african-wildlife-seg/african-wildlife-seg.yaml",  # Segmentation dataset
    epochs=80,
    patience=10,
    trainer=YOLOEPESegTrainer,  # <- Important: use segmentation trainer
    optimizer='AdamW', 
    lr0=1e-3,
    batch=8,
    workers=0,
    device=0,
    # nbs=4,          # 关键：关掉大梯度累积

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


