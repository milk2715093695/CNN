from typing import Dict, Type
from .base_model import BaseModel

class ModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, Type[BaseModel]] = {}

    def register(self, name: str) -> None:
        def decorator(cls: Type[BaseModel]):
            self._models[name] = cls
            return cls
        return decorator
        
    def get_model(self, name: str) -> Type[BaseModel]:
        if name not in self._models:
            raise ValueError(f"模型 {name} 不存在")
        return self._models[name]
    
registry = ModelRegistry()