import torch
from pathlib import Path

class Config:
    # 日志配置
    LOG_DIR = Path("./logs")
    CLEAR_LOG_ON_STARTUP = False

    # 数据配置
    DATA_DIR = Path("./data")
    DATA_BATCH_SIZE = 60

    # 模型配置
    LOAD_MODEL = False
    MODEL_LOAD_PATH = Path("./models/MNIST.pth")
    MODEL_SAVE_PATH = Path("./models/MNIST.pth")
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    NUM_EPOCHES = 10
    MAX_EPOCHES = 100
    PATIENCE = 10

    # 线程配置
    NUM_THREADS = 8

    # GUI 相关配置
    IMG_DIR = Path("./images/28_28")
    WIDTH = 1200