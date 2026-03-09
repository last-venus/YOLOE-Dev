#!/usr/bin/env python3
from pathlib import Path

# Directly set the mp4 file path here.
MP4_FILE_PATH = Path("/home/jingxiuya/YOLOE/scripts/images/20260306/0000.mp4")

# Save one image every 30 frames.
FRAME_INTERVAL = 30


def get_next_image_index(output_dir: Path) -> int:
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

    output_dir = video_path.parent / "data"
    output_dir.mkdir(exist_ok=True)
    start_index = get_next_image_index(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    saved_count = 0
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % interval == 0:
            image_path = output_dir / f"{start_index + saved_count:04d}.jpg"
            cv2.imwrite(str(image_path), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"{video_path.name}: saved {saved_count} images to {output_dir}")
    return saved_count


def main() -> None:
    if FRAME_INTERVAL <= 0:
        raise SystemExit("FRAME_INTERVAL must be a positive integer.")

    if MP4_FILE_PATH.suffix.lower() != ".mp4":
        raise SystemExit(f"Not an mp4 file: {MP4_FILE_PATH}")

    if not MP4_FILE_PATH.exists():
        raise SystemExit(f"File not found: {MP4_FILE_PATH}")

    extract_frames(MP4_FILE_PATH, FRAME_INTERVAL)


if __name__ == "__main__":
    main()
