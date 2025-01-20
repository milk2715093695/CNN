import torch
import logging

from src.utils.read_data import read_idx1_ubyte as rd1
from src.utils.read_data import read_idx3_ubyte as rd3
from src.models.train_model import train_model as train
from src.models.judge_model import judge_model as judge
from src.utils.log_config import setup_logger, clear_log_files
from src.data.transforms import transform_label, transform_data

from torch import nn, optim
from src.config import Config
from src.models.CNN import CNN_MNIST as CNN
from src.data.dataset import MNISTDataset as Dataset
from src.data.dataloader import CustomDataLoader as DataLoader

# 配置日志
if Config.CLEAR_LOG_ON_STARTUP:
    clear_log_files(Config.LOG_DIR)

setup_logger(Config.LOG_DIR)
logger = logging.getLogger()

# 配置多线程
torch.set_num_threads(Config.NUM_THREADS)
device = Config.DEVICE
logger.info(f"使用了设备 {device}\n")

def main() -> None:
    # 读取配置
    data_dir = Config.DATA_DIR
    batch_size = Config.DATA_BATCH_SIZE
    load_path = Config.MODEL_LOAD_PATH
    save_path = Config.MODEL_SAVE_PATH
    num_epoches = Config.NUM_EPOCHES

    # 初始化模型、损失函数和优化器
    model = CNN().to(device)
    
    if Config.LOAD_MODEL:
        logger.info("准备加载模型\n")
        model.load_state_dict(torch.load(load_path, weights_only=True))
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

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, timeout=0)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, timeout=0, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    loss_history = train(
        model, dataloader_train, 
        criterion=criterion, optimizer=optimizer, 
        num_epoches=num_epoches, device=device
    )

    # 保存模型到指定路径
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    torch.save(model.state_dict(), save_path)

    logger.info(f"模型已保存到: {save_path}")

    print("训练损失:", loss_history)

    acc = judge(model, dataloader_test, device=device)
    print(f"模型准确率为：{acc}")

if __name__ == "__main__":
    main()