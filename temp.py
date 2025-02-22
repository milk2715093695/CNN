import torch
from backend.core import model_registry
from backend.models.CNN.Vanilla.model import VanillaCNNConfig
from backend.core.training_strategy import DefaultTrainingStrategy

config = VanillaCNNConfig.from_yaml("configs/model_configs/CNN/Vanilla/vanilla_cnn.yaml")
cnn_model_class = model_registry.registry.get_model("vanilla_cnn")
cnn = cnn_model_class(config=config)
print(cnn)

config = {
    "optimizer": "Adam",
    "optimizer_params": {
        "lr": 0.001,
        "betas": (0.9, 0.999)
    }
}
trainer = DefaultTrainingStrategy(config=config)

optimizer = trainer.configure_optimizer(cnn)
input, target = torch.randn(5, 1, 28, 28), torch.randn(5, 10)
print(trainer.train_step(cnn, (input, target), torch.nn.CrossEntropyLoss(), optimizer))