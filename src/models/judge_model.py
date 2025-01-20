import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader

def judge_model(
    model: nn.Module,
    data_loader: DataLoader,
    /,
    device: torch.device = "cpu"
) -> float:
    # 配置日志
    logger = logging.getLogger()
    logger.info("开始评估\n")

    model = model.to(device)
    model.eval()  # 切换为评估模式

    correct = 0
    total = 0

    with torch.no_grad():  # 评估时不需要梯度计算
        for batch_idx, (labels, images) in enumerate(data_loader):
            labels, images = labels.to(device), images.to(device)

            # 前向传播
            outputs = model(images)

            # 计算预测结果
            _, predicted = torch.max(outputs, 1)
            _, actual = torch.max(labels, 1)

            # 更新统计信息
            total += labels.size(0)
            correct += (predicted == actual).sum().item()

    # 计算准确率
    accuracy = 100 * correct / total
    logger.info(f"评估完成，准确率: {accuracy:.2f}%\n")

    return accuracy