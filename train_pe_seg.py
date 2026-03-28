import sys
sys.path.insert(0, "/home/jingxiuya/YOLOE-Dev")

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

# Load a pretrained segmentation model
model = YOLOE("/home/jingxiuya/datasets/pt/yoloe-26s-seg.pt")

# Identify the head layer index
head_index = len(model.model.model) - 1

# Freeze all backbone and neck layers (i.e., everything before the head)
freeze = [str(i) for i in range(0, head_index)]

# Freeze parts of the segmentation head, keeping only the classification branch trainable
for name, child in model.model.model[-1].named_children():
    if "cv3" not in name:
        freeze.append(f"{head_index}.{name}")

# Freeze detection branch components
freeze.extend(
    [
        f"{head_index}.cv3.0.0",
        f"{head_index}.cv3.0.1",
        f"{head_index}.cv3.1.0",
        f"{head_index}.cv3.1.1",
        f"{head_index}.cv3.2.0",
        f"{head_index}.cv3.2.1",
    ]
)

# Train only the classification branch
results = model.train(
    data="/home/jingxiuya/datasets/african-wildlife-seg-overfit12/african-wildlife-seg-overfit12.yaml",  # Segmentation dataset
    epochs=40,
    patience=10,
    trainer=YOLOEPESegTrainer,   # <- Important: use detection trainer
    optimizer='AdamW', 
    lr0=1e-3,
    freeze=freeze,
    batch=16,
    workers=0,
    device=0,
)
