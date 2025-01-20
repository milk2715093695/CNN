import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.read_data import read_idx3_ubyte as rd3

def display(image: torch.Tensor) -> None:
    image = image.float() / 255.0

    # 创建 RGB 网格图像，预留网格线的像素，| 6图 | 6图 | 6图 | ... | 6图 | 
    grid_size = 28 * 7 + 1
    grid_image = np.ones((grid_size, grid_size, 3))  # 初始化为白色背景 (1, 1, 1)

    # 填充图像像素（灰度值）
    for i in range(28):
        for j in range(28):
            pixel_value = image[i, j].item()
            grid_image[i * 7 + 1: i * 7 + 7, j * 7 + 1: j * 7 + 7] = [pixel_value, pixel_value, pixel_value]  # 灰度值填充

    # 在网格线部分设置为蓝色 (0, 0, 1)
    grid_image[::7, :, :] = [0, 0, 1]  # 水平网格线
    grid_image[:, ::7, :] = [0, 0, 1]  # 垂直网格线

    # 绘制图像
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_image, vmin=0, vmax=1)  # RGB 模式直接显示
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images = rd3(Path("./data/train/train-images-idx3-ubyte"), num=100)

    num = 5
    for i in range(num):
        image = images[i][0]
        display(image)


