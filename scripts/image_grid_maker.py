from math import ceil
from pathlib import Path

import cv2
import numpy as np


class ImageGridMaker:
    """将目录中的图片拼接为一张网格总览图。"""

    def __init__(
        self,
        image_dir: str,
        rows: int = 5,
        cols: int = 5,
        output_name: str = "image_grid.jpg",
    ) -> None:
        # 输入目录、网格行列数和输出文件名都在初始化时配置好，方便外部直接复用这个类。
        self.image_dir = Path(image_dir)
        self.rows = rows
        self.cols = cols
        self.output_name = output_name
        self.supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def collect_images(self) -> list[Path]:
        """读取目录下文件名以数字开头的图片，并按文件名排序。"""
        if not self.image_dir.exists() or not self.image_dir.is_dir():
            raise NotADirectoryError(f"Invalid image directory: {self.image_dir}")

        image_paths = []
        for image_path in self.image_dir.iterdir():
            # 只保留普通文件、支持的图片格式，且文件名必须以数字开头，例如 0000.jpg。
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in self.supported_suffixes:
                continue
            if not image_path.stem or not image_path.stem.isdigit():
                continue
            image_paths.append(image_path)

        image_paths.sort()
        if not image_paths:
            raise FileNotFoundError(
                f"No numeric-leading images found in: {self.image_dir}"
            )

        return image_paths

    def load_images(self, image_paths: list[Path]) -> list[np.ndarray]:
        """加载图片；如果尺寸不一致，则统一缩放到第一张图片的尺寸。"""
        images = []
        base_size = None

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skip unreadable image: {image_path}")
                continue

            if base_size is None:
                # 第一张图的尺寸作为所有网格单元的基准尺寸。
                base_size = (image.shape[1], image.shape[0])
            elif (image.shape[1], image.shape[0]) != base_size:
                # 后续图片如果尺寸不同，统一缩放后再拼接，避免网格错位。
                image = cv2.resize(image, base_size)

            images.append(image)

        if not images:
            raise FileNotFoundError("No readable images were found.")

        return images

    def make_grid(self) -> Path:
        """将图片按网格拼接并保存到目录下。"""
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive integers.")

        image_paths = self.collect_images()
        images = self.load_images(image_paths)

        cell_h, cell_w = images[0].shape[:2]
        total_images = len(images)

        # 如果图片数量超过预设网格容量，则自动增加行数，保证所有图片都能放下。
        actual_rows = max(self.rows, ceil(total_images / self.cols))
        # 先创建一张白色背景的大图，再把每张小图依次贴到对应位置。
        grid = np.full(
            (actual_rows * cell_h, self.cols * cell_w, 3),
            255,
            dtype=np.uint8,
        )

        for index, image in enumerate(images):
            # 根据当前图片下标计算它应该放在第几行、第几列。
            row = index // self.cols
            col = index % self.cols
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            grid[y1:y2, x1:x2] = image

        output_path = self.image_dir / self.output_name
        cv2.imwrite(str(output_path), grid)
        print(f"Grid image saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    # 直接运行本文件时，会使用下面的示例参数生成一张拼图总览。
    maker = ImageGridMaker(
        image_dir="/home/jingxiuya/YOLOE/scripts/runs/260310/0000_26s_paper_and_red_cups",
        rows=5,
        cols=5,
        output_name="image_grid.jpg",
    )
    maker.make_grid()
