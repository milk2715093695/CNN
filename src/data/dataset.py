import torch
import logging

from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple

class MNISTDataset(Dataset):
    def __init__(
        self,
        root_dir: Path = Path("./data"),
        split: str = "train",
        filename_label: Optional[str] = None,
        filename_data: Optional[str] = None,
        *,
        label_reader: Callable[[Path, Optional[int], torch.device], torch.Tensor],
        data_reader: Callable[[Path, Optional[int], torch.device], torch.Tensor],
        num: Optional[int] = None,
        transform_label: Optional[Callable[[torch.Tensor, torch.device], torch.Tensor]] = None,
        transform_data: Optional[Callable[[torch.Tensor, torch.device], torch.Tensor]] = None,
        device: torch.device = "cpu"
    ) -> None:
        """
        数据集初始化

        Args:
            label_reader (Callable[[Path, Optional[int], torch.device], torch.Tensor]): 读取 label 的函数。
            data_reader (Callable[[Path, Optional[int], torch.device], torch.Tensor]): 读取 data 的函数。
            root_dir (Path, optional): 数据目录。默认为 Path("./data")。
            split (str, optional): 使用的数据集，默认为 "train"。
            filename_label (Optional[str], optional): label 数据的文件名，默认为 None。
            filename_data (Optional[str], optional): data 数据的文件名，默认为 None。
            num (Optional[int], optional): 读取的数据数量，默认为 None。
            transform_label (Optional[Callable[[torch.Tensor, torch.device], torch.Tensor]]): 对 label 数据进行的变换，默认为 None。
            transform_data (Optional[Callable[[torch.Tensor, torch.device], torch.Tensor]]): 对 data 数据进行的变换，默认为 None。
            device (Optional[torch.device]): 使用的设备，默认为 "cpu"。
        """
        # 获取日志
        self.logger = logging.getLogger()

        # 参数校验
        if not (root_dir.exists() and root_dir.is_dir()):
            err_info = f"目录 {root_dir} 不存在！\n"
            self.logger.error(err_info)
            raise ValueError(err_info)

        # 默认值处理
        prefix = "train" if split == "train" else "t10k"
        filename_label = f"{prefix}-labels-idx1-ubyte" if filename_label is None else filename_label
        filename_data = f"{prefix}-images-idx3-ubyte" if filename_data is None else filename_data
        
        # 获取路径
        label_path = root_dir / split / filename_label
        data_path = root_dir / split / filename_data
        
        # 解析数据
        self.labels = label_reader(label_path, num=num, device=device)
        self.datas = data_reader(data_path, num=num, device=device)

        # 数据类型校验
        if not isinstance(self.labels, torch.Tensor):
            err_info = (
                "数据类型错误！\n"
                f"期望的数据类型为 torch.tensor，实际 labels 的类型为 {type(self.labels)}\n"
            )
            self.logger.error(err_info)
            raise TypeError(err_info)
        if not isinstance(self.datas, torch.Tensor):
            err_info = (
                "数据类型错误！\n"
                f"期望的数据类型为 torch.tensor，实际 datas 的类型为 {type(self.datas)}\n"
            )
            self.logger.error(err_info)
            raise TypeError(err_info)

        # 应用变换
        self.labels = transform_label(self.labels, device=device) if transform_label is not None else self.labels
        self.datas = transform_data(self.datas, device=device) if transform_data is not None else self.datas

        # 数据类型校验
        if not isinstance(self.labels, torch.Tensor):
            err_info = (
                "数据类型错误！\n"
                f"期望的数据类型为 torch.tensor，实际 labels 转换后的类型为 {type(self.labels)}\n"
            )
            self.logger.error(err_info)
            raise TypeError(err_info)
        if not isinstance(self.datas, torch.Tensor):
            err_info = (
                "数据类型错误！\n"
                f"期望的数据类型为 torch.tensor，实际 datas 转换后的类型为 {type(self.datas)}\n"
            )
            self.logger.error(err_info)
            raise TypeError(err_info)


        # 数据校验
        if self.labels.shape[0] != self.datas.shape[0]:
            err_info = (
                "labels 的长度与 datas 的长度不匹配！\n"
                f"labels 的长度为 {self.labels.shape[0]}，datas 的长度为 {self.datas.shape[0]}！\n"
            )
            self.logger.error(err_info)
            raise ValueError(err_info)

        self.logger.info(
            "成功建立数据集！\n"
            f"数据文件：{label_path} 与 {data_path}\n"
            f"labels 的形状：{self.labels.shape}，datas 的形状：{self.datas.shape}\n"
        )

    def __len__(self) -> int:
        """返回数据集的长度"""
        return self.labels.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= idx < len(self) and idx == int(idx)):
            err_info = (
                "下标越界！\n"
                f"正确长度为 [0, {len(self) - 1}]，实际访问了 {idx}！\n"
            )
            self.logger.error(err_info)
            raise IndexError(err_info)
        return self.labels[idx], self.datas[idx]
    