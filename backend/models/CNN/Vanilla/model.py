import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from backend.core import base_model, model_registry


@dataclass
class VanillaCNNConfig(base_model.BaseModelConfig):
    input_shape: List[int] = None
    conv_layers: List[Dict[str, int]] = None
    fc_layers: List[Dict[str, int]] = None
    dropout_rate: float = 0.5

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "VanillaCNNConfig":
        """ 从 YAML 文件加载配置并返回 VanillaCNNConfig 对象 """
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        params = yaml_data["model"]["params"]
        return cls(
            input_shape=params["input_shape"],
            conv_layers=params["conv_layers"],
            fc_layers=params["fc_layers"],
            dropout_rate=params["dropout_rate"]
        )


@model_registry.registry.register("vanilla_cnn")
class VanillaCNN(base_model.BaseModel):
    def _check_config(self) -> VanillaCNNConfig:
        config = self.config
        if not isinstance(config, VanillaCNNConfig):
            raise TypeError(f"配置类型错误！预期 VanillaCNNConfig 类型，实际 {type(config)} 类型")
        if config.input_shape is None or config.conv_layers is None or config.fc_layers is None:
            raise ValueError("配置中的 input_shape、conv_layers 或 fc_layers 不能为空")
        return config

    def _build_layers(self) -> None:
        config = self._check_config()
        channels, height, width = config.input_shape

        # 创建卷积层
        self.conv_layers = nn.ModuleList()
        for layer_config in config.conv_layers:
            kernel_size = layer_config["kernel_size"]
            stride = layer_config["stride"]
            padding = layer_config["padding"]

            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=channels, 
                    out_channels=layer_config["out_channels"],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )

            # 更新各个值
            channels = layer_config['out_channels']
            height = (height + 2 * padding - kernel_size) // stride + 1
            height //= 2
            width = (width + 2 * padding - kernel_size) // stride + 1
            width //= 2
        
        units = channels * height * width
        self.flat_features = units

        # 创建全连接层
        self.fc_layers = nn.ModuleList()
        for layer_config in config.fc_layers:
            self.fc_layers.append(
                nn.Linear(units, layer_config['units'])
            )
            units = layer_config['units']

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        
        for conv_layer, layer_config in zip(self.conv_layers, self.config.conv_layers):
            x = conv_layer(x)
            if layer_config.get("activate", None):
                x = self._get_activation_function(layer_config["activate"])(x)
            x = F.max_pool2d(x, 2, 2)

        # 展平
        x = x.view(-1, self.flat_features)

        # 通过全连接层
        for fc_layer, layer_config in zip(self.fc_layers, self.config.fc_layers):
            x = fc_layer(x)
            if layer_config.get("activate", None):
                x = self._get_activation_function(layer_config["activate"])(x)
            x = self.dropout(x)

        return x
    

if __name__ == "__main__":
    config = VanillaCNNConfig.from_yaml("configs/model_configs/CNN/Vanilla/vanilla_cnn.yaml")

    cnn = VanillaCNN(config)
    print(cnn)

    test_input = torch.randn((5, 1, 28, 28))
    print(cnn(test_input))