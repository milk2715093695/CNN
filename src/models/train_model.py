import torch
import logging
import torch.nn as nn

from typing import List
from torch.utils.data import DataLoader

def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    /,
    criterion: nn.Module,
    optimizer: nn.Module,
    num_epoches: int,
    device: torch.device = "cpu",
) -> List[float]:
    # 配置日志
    logger = logging.getLogger()
    logger.info("开始训练\n")
    
    model = model.to(device)
    loss_values = []

    for epoch in range(num_epoches):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (labels, images) in enumerate(data_loader):
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
        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_values.append(avg_epoch_loss)

        logger.info(f"第 {epoch + 1} 轮训练完成，平均损失：{avg_epoch_loss: .5f}\n")
        
    logger.info("训练完成\n")
    return loss_values