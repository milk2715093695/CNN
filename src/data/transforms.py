import torch
import logging

from torchvision import transforms

def transform_label(label: torch.Tensor, *, device: torch.device = "cpu") -> torch.Tensor:
    """利用 `scatter` 将标签转为独热编码"""
    # 配置日志
    logger = logging.getLogger()

    # 创建全为 0 的空编码张量
    one_hot = torch.zeros(label.shape[0], 10, device=device)

    # 转为独热编码
    one_hot.scatter_(dim=1, index=label.view(-1, 1), value=1)
    logger.debug(
        "转换 label 完成\n"
        f"转换前形状 {label.shape}，转换后形状 {one_hot.shape}\n"
    )
    return one_hot.float()

def transform_data(data: torch.Tensor, *, device: torch.device = "cpu") -> torch.Tensor:
    # 配置日志
    logger = logging.getLogger()

    data = data.float()

    # 标准化
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

    channels = data.shape[1]
    normalized_channels = [
        normalize(data[:, c, :, :]) for c in range(channels)
    ]
    normalized_data = torch.stack(normalized_channels, dim=1)  # 合并回原始形状

    assert(isinstance(normalized_data, torch.Tensor))
    normalized_data = normalized_data.to(device)

    logger.debug(
        "转换 data 完成\n"
        f"转换前形状 {data.shape}，转换后形状 {normalized_data.shape}\n"
    )

    return normalized_data