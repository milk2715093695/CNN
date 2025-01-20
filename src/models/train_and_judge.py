import torch
import logging
import torch.nn as nn
from typing import List, Callable
from torch.utils.data import DataLoader

def train_and_judge(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    /,
    judge_model: Callable[[nn.Module, DataLoader, torch.device], float],
    criterion: nn.Module,
    optimizer: nn.Module,
    num_epoches: int,
    device: torch.device = "cpu",
    patience: int = 3,  # 连续多少个epoch没有提升则停止训练
    delta: float = 0.01  # 每次提升的准确率阈值
) -> List[float]:
    # 配置日志
    logger = logging.getLogger()
    logger.info("开始训练和评估\n")

    model = model.to(device)
    loss_values = []
    accuracy_values = []

    best_accuracy = 0.0  # 记录最佳准确率
    epochs_without_improvement = 0  # 记录没有提升的epoch数

    for epoch in range(num_epoches):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (labels, images) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            logger.debug(f"第 {epoch + 1} 轮，第 {batch_idx + 1} 个 batch 完成，损失：{loss: .5f}\n")

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累加每个批次的损失
            epoch_loss += loss.item()

        # 计算当前 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_epoch_loss)

        logger.info(f"第 {epoch + 1} 轮训练完成，平均损失：{avg_epoch_loss: .5f}\n")

        # 在每个 epoch 结束后进行评估
        accuracy = judge_model(model, test_loader, device)
        accuracy_values.append(accuracy)

        logger.info(f"第 {epoch + 1} 轮评估完成，准确率：{accuracy:.2f}%\n")

        # 检查是否有提升
        if accuracy > best_accuracy + delta:
            best_accuracy = accuracy
            epochs_without_improvement = 0  # 重置
        else:
            epochs_without_improvement += 1

        # 如果超过耐心值，停止训练
        if epochs_without_improvement >= patience:
            logger.info(f"准确率没有提升，训练提前结束（已连续 {patience} 个epoch没有提升）\n")
            break

    logger.info("训练和评估完成\n")
    return loss_values, accuracy_values