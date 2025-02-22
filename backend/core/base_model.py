import torch
import logging
import torch.nn as nn
from enum import Enum
from pathlib import Path
import torch.nn.functional as F
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union, Optional, Any


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义配置类型泛型
ConfigType = TypeVar("ConfigType", bound="BaseModelConfig")


class DeviceType(Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class BaseModelConfig:
    """模型配置基类

    Attributes:
        device (DeviceType): 使用的设备类型，支持自动选择(AUTO)、CPU、CUDA、MPS
        strict_device_validation (bool): 是否严格验证设备可用性，如果设为True且指定设备不可用时抛出异常
    """
    device: DeviceType = DeviceType.AUTO
    strict_device_validation: bool = True   

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({fields})"


class BaseModel(ABC, nn.Module, Generic[ConfigType]):
    """所有模型的抽象基类，定义统一接口

    Attributes:
        config (ConfigType): 模型配置对象
        device (torch.device): 模型当前运行的设备
    """

    def __init__(self, config: ConfigType) -> None:
        super().__init__()

        self.config = config
        self.device: torch.device = torch.device("cpu")  # 初始化为CPU，后续会重置

        self._build_layers()
        self._set_device()

    def _set_device(self) -> None:
        """根据配置设置设备并移动模型"""
        if self.config.device == DeviceType.AUTO:
            device = self._auto_select_device()
        else:
            device = torch.device(self.config.device.value)
            if self.config.strict_device_validation:
                self._validate_device_availability(device)

        try:
            self._validate_device_availability(device)
        except RuntimeError as e:
            if self.config.strict_device_validation:
                raise
            else:
                logger.warning(f"{e}，自动回退到CPU")
                device = torch.device("cpu")

        self.device = device
        self.to(self.device)
        logger.info(f"模型运行在设备：{self.device}")

    def _auto_select_device(self) -> torch.device:
        """自动选择设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except AttributeError:
            pass

        return torch.device("cpu")
    
    def _validate_device_availability(self, device: torch.device) -> None:
        """验证设备是否可用"""
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求的CUDA设备不可用，当前PyTorch不支持CUDA")
        if device.type == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError("请求的MPS设备不可用，当前PyTorch不支持MPS")

    def save_model(self, path: Union[Path, str]) -> None:
        """保存模型权重和配置到指定路径"""
        path = Path(path) if isinstance(path, str) else path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'state_dict': self.state_dict(),
                'config': self.config
            }, path)
            logger.info(f"模型成功保存到路径：{path}")
        except Exception as e:
            logger.error(f"模型保存失败：{str(e)}")
            raise IOError(f"无法保存模型到 {path}") from e
        
    @classmethod
    def load_model(
        cls,
        path: Union[Path, str],
        config: Optional[ConfigType] = None,
        **kwargs: Any
    ) -> "BaseModel":
        """加载模型权重和配置"""
        path = Path(path) if isinstance(path, str) else path
        if not path.exists():
            raise FileNotFoundError(f"权重文件 {path} 不存在")

        try:
            checkpoint = torch.load(
                path,
                map_location=torch.device('cpu'),  # 先加载到CPU防止设备冲突
                weights_only=False,
                **kwargs
            )
            
            # 优先使用传入的配置，其次使用保存的配置
            loaded_config = config if config else checkpoint['config']
            model = cls(loaded_config)
        
            # 移动模型到正确设备
            model.to(model.device)
            
            # 加载权重并处理设备
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"从路径 {path} 成功加载模型")
            return model
        except Exception as e:
            logger.error(f"模型加载失败：{str(e)}")
            raise IOError(f"无法从 {path} 加载模型") from e
        
    def _get_activation_function(self, activate: str):
        """ 返回指定的激活函数 """
        if activate == "relu":
            return F.relu
        elif activate == "sigmoid":
            return torch.sigmoid
        elif activate == "tanh":
            return torch.tanh
        elif activate is None or activate == "None":
            return lambda x: x
        else:
            raise ValueError(f"不支持的激活函数: {activate}")

    @abstractmethod
    def _build_layers(self) -> None:
        """网络层构建方法（必须由子类实现）"""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """基类前向传播（子类需调用super().forward(x)）"""
        if x.device != self.device:
            x = x.to(self.device)
            logger.debug(f"自动将输入数据转移到设备: {self.device}")
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"
    

if __name__ == "__main__":
    @dataclass
    class ExampleConfig(BaseModelConfig):
        input_size: int = 784
        hidden_size: int = 256
        output_size: int = 10

    class ExampleModel(BaseModel[ExampleConfig]):
        def _build_layers(self) -> None:
            self.layers = nn.Sequential(
                nn.Linear(self.config.input_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, self.config.output_size)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = super().forward(x)  # 调用设备转移
            return self.layers(x)

    # 初始化配置和模型
    config = ExampleConfig(device=DeviceType.AUTO)
    model = ExampleModel(config)

    # 测试前向传播
    dummy_input = torch.randn(32, 784)
    output = model(dummy_input)
    logger.info(f"示例模型输出形状：{output.shape}")

    # 测试保存/加载
    model.save_model("example_model.pth")
    loaded_model = ExampleModel.load_model("example_model.pth", None)

    output = loaded_model(dummy_input)
    logger.info(f"加载的模型输出形状：{output.shape}")