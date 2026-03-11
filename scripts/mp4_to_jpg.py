#!/usr/bin/env python3
from pathlib import Path

# 直接在这里填写需要处理的 mp4 文件路径
MP4_FILE_PATH = Path("/home/jingxiuya/YOLOE/scripts/images/20260311/0000.mp4")

# 每隔 30 帧保存一张图片
FRAME_INTERVAL = 45


def get_next_image_index(output_dir: Path) -> int:
    # 扫描输出目录中现有的 jpg 文件，找到下一个可用编号
    max_index = -1
    for image_path in output_dir.glob("*.jpg"):
        if image_path.stem.isdigit():
            max_index = max(max_index, int(image_path.stem))
    return max_index + 1


def extract_frames(video_path: Path, interval: int) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required. Please install it first: pip install opencv-python"
        ) from exc

    # 在视频同级目录下创建 data 文件夹，用来保存导出的图片
    output_dir = video_path.parent / "data"
    output_dir.mkdir(exist_ok=True)
    start_index = get_next_image_index(output_dir)

    # 打开视频文件，准备逐帧读取
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    saved_count = 0
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            # 读不到新帧时说明视频已经处理结束
            break

        if frame_index % interval == 0:
            # 按 0000.jpg、0001.jpg 这样的格式保存图片
            image_path = output_dir / f"{start_index + saved_count:04d}.jpg"
            cv2.imwrite(str(image_path), frame)
            saved_count += 1

        # 继续处理下一帧
        frame_index += 1

    cap.release()
    print(f"{video_path.name}: saved {saved_count} images to {output_dir}")
    return saved_count


def main() -> None:
    # 检查抽帧间隔是否合法
    if FRAME_INTERVAL <= 0:
        raise SystemExit("FRAME_INTERVAL must be a positive integer.")

    # 检查文件后缀是否为 mp4
    if MP4_FILE_PATH.suffix.lower() != ".mp4":
        raise SystemExit(f"Not an mp4 file: {MP4_FILE_PATH}")

    # 检查文件是否真实存在
    if not MP4_FILE_PATH.exists():
        raise SystemExit(f"File not found: {MP4_FILE_PATH}")

    # 开始执行抽帧
    extract_frames(MP4_FILE_PATH, FRAME_INTERVAL)


if __name__ == "__main__":
    main()
