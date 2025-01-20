import torch
import logging

from pathlib import Path
from struct import unpack
from typing import Optional

def read_idx1_ubyte(path: Path, /, num: Optional[int] = None, *, device: torch.device = "cpu") -> torch.Tensor:
    """
    读取 IDX1 文件中的标签数据并返回张量。

    Args:
        path (Path): 文件路径，指向 IDX1 文件。
        num (Optional[int], optional): 读取的标签数量。如果为 None，读取全部。
        device (torch.device): 计算的设备信息。默认为 "cpu"。

    Raises:
        FileNotFoundError: 如果文件不存在或者不是文件。
        ValueError: 如果文件魔数不正确。
        ValueError: 如果读取的数量不正确。
        ValueError: 如果文件内容不完整。

    Returns:
        torch.Tensor: 标签的张量数据。
    """
    # 获取日志
    logger = logging.getLogger()

    # 参数验证
    if not path.exists() or not path.is_file():
        error_info = f"文件 {path} 不存在！\n"
        logger.error(error_info)
        raise FileNotFoundError(error_info)

    with path.open('rb') as file:
        header = file.read(8)                       # 前 8 个字节为数据头
        magic, num_labels = unpack(">II", header)   # > 表示大端序，每个 I 表示一个 unsigned int
        num = num if num is not None else num_labels

        # 文件校验
        if magic != 2049:
            error_info = (
                f"文件 {path} 解析异常，这可能不是 IDX1 文件！\n"
                f"正确的魔数为 2049，实际上为 {magic}！\n"
            )
            logger.error(error_info)
            raise ValueError(error_info)
        # 参数校验
        if num > num_labels or num < 0:
            error_info = (
                "读取的数量不正确！\n"
                f"正确范围：[0, {num_labels}]，实际传入 {num}！\n"
            )
            logger.error(error_info)
            raise ValueError(error_info)
        
        label_data = file.read(num)
        if len(label_data) != num:
            error_info = (
                "文件内容不完整！\n"
                f"预计有 {num_labels} 个字节，实际上读到 {len(label_data)} 字节！\n"
            )
            logger.error(error_info)
            raise ValueError(error_info)
        
        logger.info(f"成功从 {path} 中读取 {num} 个数据！\n")
        return torch.tensor(list(label_data), dtype=torch.int64, device=device).unsqueeze(1)
    
def read_idx3_ubyte(path: Path, /, num: Optional[int] = None, *, device: torch.device = "cpu") -> torch.Tensor:
    """
    读取 IDX3 文件中的标签数据并返回张量。

    Args:
        path (Path): 文件路径，指向 IDX3 文件。
        num (Optional[int], optional): 读取的图像数量。如果为 None，读取全部。
        device (torch.device): 计算的设备信息。默认为 "cpu"。

    Raises:
        FileNotFoundError: 如果文件不存在或者不是文件。
        ValueError: 如果文件魔数不正确。
        ValueError: 如果读取的数量不正确。
        ValueError: 如果文件内容不完整。

    Returns:
        torch.Tensor: 图像的张量数据。
    """
    # 获取日志
    logger = logging.getLogger()

    # 参数验证
    if not path.exists() or not path.is_file():
        error_info = f"文件 {path} 不存在！\n"
        logger.error(error_info)
        raise FileNotFoundError(error_info)
    
    with path.open('rb') as file:
        header = file.read(16)
        magic, num_images, num_rows, num_cols = unpack('>IIII', header)
        num = num if num is not None else num_images

        # 文件校验
        if magic != 2051:
            error_info = (
                f"文件 {path} 解析异常，这可能不是 IDX3 文件！\n"
                f"正确的魔数为 2051，实际上为 {magic}！\n"
            )
            logger.error(error_info)
            raise ValueError(error_info)
        # 参数校验
        if num > num_images or num < 0:
            error_info = (
                "读取的数量不正确！\n"
                f"正确范围：[0, {num_images}]，实际传入 {num}！\n"
            )
            logger.error(error_info)
            raise ValueError(error_info)
        
        image_data = file.read(num * num_rows * num_cols)
        if len(image_data) != num * num_rows * num_cols:
            error_info = (
                "文件内容不完整！\n"
                f"预计有 {num_images * num_rows * num_cols} 字节，实际上读到 {len(image_data)} 字节！\n"
            )
            logger.error(error_info)
            raise ValueError(error_info)
        
        images = torch.tensor(list(image_data), dtype=torch.uint8, device=device)
        images = images.view(num, 1, num_rows, num_cols)

        logger.info(f"成功从 {path} 读取了 {num} 张大小为 {num_rows} * {num_cols} 的图像！\n")
        return images