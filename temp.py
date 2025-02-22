from backend.core import model_registry
from backend.models.CNN.Vanilla.model import VanillaCNNConfig

config = VanillaCNNConfig.from_yaml("configs/model_configs/CNN/Vanilla/vanilla_cnn.yaml")
cnn_model_class = model_registry.registry.get_model("vanilla_cnn")
cnn = cnn_model_class(config=config)
print(cnn)