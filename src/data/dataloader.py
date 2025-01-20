import torch
import logging

from typing import Iterator, Tuple
from torch.utils.data import Dataset, DataLoader

class CustomDataLoader(DataLoader):
    def __init__(
            self, 
            dataset: Dataset,
            /,
            batch_size: int = 1,
            shuffle: bool = True,
            num_workers: int = 0,
            drop_last: bool = False,
            timeout: float = 30,
    ) -> None:
        """初始化数据加载器

        Args:
            dataset (CIFAR10Dataset): 数据集
            batch (int, optional): 每一个批次的大小。默认为 1。
            shuffle (bool, optional): 每一次后是否打乱。默认为 True。
            num_workers (int, optional): 使用的线程数量。默认为 0。
            drop_last (bool, optional): 是否抛弃多余的样本。默认为 False。
            timeout (float, optional): 超时时间。默认为 30。
        """
        # 配置日志
        self.logger = logging.getLogger()

        self.shuffle = shuffle
        self.batch_size = batch_size
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            drop_last=drop_last, 
            timeout=timeout
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """重写迭代器，在每次迭代时打印基本信息

        Yields:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]: 数据与标签
        """
        self.logger.debug(f"批次大小: {self.batch_size}，是否打乱: {'是' if self.shuffle else '否'}")
        
        for i, (label, data) in enumerate(super().__iter__()):
            self.logger.debug(f"第 {i + 1} 个批次的 labels 形状为 {label.shape}，datas 形状为 {data.shape}")
            yield label, data
        
        self.logger.info("数据加载完成！\n")