import torch
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod

class BaseTrainingStrategy(ABC):
    """抽象基类，定义了训练策略的基本结构。所有具体训练策略必须继承此类并实现其方法。"""
    
    @abstractmethod
    def configure_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """配置并返回优化器。"""
        pass

    @abstractmethod
    def train_step(
        self, 
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """执行单步训练操作，包括前向传播、损失计算和反向传播。"""
        pass

    @abstractmethod
    def validation_step(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: torch.nn.Module
    ) -> Tuple[float, int]:
        """执行单步验证操作，包括前向传播和损失计算。"""
        pass

class DefaultTrainingStrategy(BaseTrainingStrategy):
    """默认训练策略"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def configure_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = getattr(torch.optim, self.config["optimizer"])(
            model.parameters(),
            **self.config["optimizer_params"]
        )
        return optimizer
    
    def train_step(
        self, 
        model: torch.nn.Module, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer
    ) -> float:
        inputs, targets = batch
        inputs, targets = inputs.to(model.device), targets.to(model.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        return loss.item()
    
    def validation_step(
        self, 
        model: torch.nn.Module, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        criterion: torch.nn.Module
    ) -> Tuple[float, int]:
        inputs, targets = batch
        inputs, targets = inputs.to(model.device), targets.to(model.device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            correct = (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

        return loss.item(), correct