import torch
import logging

from src.utils.read_data import read_idx1_ubyte as rd1
from src.utils.read_data import read_idx3_ubyte as rd3
from src.models.train_model import train_model as train
from src.models.judge_model import judge_model as judge
from src.utils.log_config import setup_logger, clear_log_files
from src.data.transforms import transform_label, transform_data

from pathlib import Path
from torch import nn, optim
from src.models.CNN import CNN_MNIST as CNN
from src.data.dataset import MNISTDataset as Dataset
from src.data.dataloader import CustomDataLoader as DataLoader

# 配置日志
clear_log = input("是否清空日志？\n>>>")
if clear_log.lower() in {'y', "yes", '1', "true"}:
    clear_log_files(Path("./logs"))

setup_logger(Path("./logs"))
logger = logging.getLogger()

# 配置多线程
torch.set_num_threads(8)
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"使用了设备 {device}")

def main() -> None:
    data_dir = Path("./data")

    # 初始化模型、损失函数和优化器
    model = CNN().to(device)
    
    load_model = input("是否加载模型？\n>>>")
    if load_model.lower() in {'y', "yes", '1', "true"}:
        logger.info("准备加载模型\n")
        model.load_state_dict(torch.load("models/MNIST.pth", weights_only=True))
        logger.info("模型加载成功！\n")
    else:
        logger.info("不加载模型\n")

    dataset_train = Dataset(
        data_dir, 
        "train", 
        label_reader=rd1, data_reader=rd3, 
        transform_label=transform_label, transform_data=transform_data,
        device=device
    )
    dataset_test = Dataset(
        data_dir, 
        "test", 
        label_reader=rd1, data_reader=rd3, 
        transform_label=transform_label, transform_data=transform_data,
        device=device
    )

    dataloader_train = DataLoader(dataset_train, batch_size=60, timeout=0)
    dataloader_test = DataLoader(dataset_test, batch_size=60, timeout=0, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    loss_history = train(model, dataloader_train, criterion=criterion, optimizer=optimizer, num_epoches=5, device=device)

    # 保存模型到指定路径
    save_path = Path("./models/MNIST.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    torch.save(model.state_dict(), save_path)

    logger.info(f"模型已保存到: {save_path}")

    print("训练损失:", loss_history)

    acc = judge(model, dataloader_test, device=device)
    print(f"模型准确率为：{acc}")

if __name__ == "__main__":
    main()