from pathlib import Path

import cv2
from ultralytics import YOLOE
 
# 定义一个包含希望检测的目标类别的列表
# names = ["robot", "bottle", "cup", "Faucet"]
names = ["Cup"]
# 加载预训练的 YOLOE 分割模型权重文件
model = YOLOE("/home/jingxiuya/YOLOE/pt/yoloe-26l-seg.pt") 
 
# 为模型设置要检测的类别
# get_text_pe(names) 会根据类别名称生成文本嵌入（text prompt embeddings）
model.set_classes(names, model.get_text_pe(names))

IMAGE = "/home/jingxiuya/YOLOE/scripts/images/20250310/data/image_grid.jpg"  # 替换为你的输入图像路径

# 进行预测，并保存结果
results = model.predict(
    source=IMAGE,
    save=True,
    project="/home/jingxiuya/YOLOE/scripts/runs/260306",
    name="cup_detection",
    exist_ok=True,
)


def draw_box_indices(result, output_path: Path) -> None:
    plotted = result.plot()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        cv2.imwrite(str(output_path), plotted)
        return

    for i, box in enumerate(boxes, start=1):
        x1, y1, _, _ = map(int, box.xyxy[0].tolist())
        label = f"#{i}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        text_x = max(0, x1)
        text_y = max(text_h + 10, y1 - 10)
        cv2.rectangle(
            plotted,
            (text_x, text_y - text_h - baseline - 8),
            (text_x + text_w + 12, text_y + 4),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            plotted,
            label,
            (text_x + 6, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), plotted)

# 打印检测框坐标、类别和置信度
for result in results:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("No detections.")
        continue

    output_path = Path(result.save_dir) / Path(result.path).name
    draw_box_indices(result, output_path)

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        cls_name = result.names.get(cls_id, str(cls_id))
        print(
            f"Box {i}: cls={cls_name}, conf={conf:.3f}, "
            f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

# 在实时窗口中显示预测结果(可选)
# results[0].show()
