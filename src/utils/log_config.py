import logging 

from pathlib import Path

class InfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.INFO

def setup_logger(log_dir: Path = Path("./logs")) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 完整日志
    full_handler = logging.FileHandler(log_dir / "full.log", mode='a', encoding="utf-8")
    full_handler.setLevel(logging.DEBUG)
    full_handler.setFormatter(formatter)

    # 错误日志
    error_handler = logging.FileHandler(log_dir / "error.log", mode='a', encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # INFO 日志
    info_handler = logging.FileHandler(log_dir / "info.log", mode='a', encoding="utf-8")
    info_handler.setLevel(logging.DEBUG)
    info_handler.setFormatter(formatter)
    info_handler.addFilter(InfoFilter())

    # 添加到根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(full_handler)
    logger.addHandler(error_handler)
    logger.addHandler(info_handler)

def clear_log_files(log_dir: Path = Path("./logs")) -> None:
    if log_dir.exists() and log_dir.is_dir():
        for log_file in log_dir.glob("*.log"):
            try:
                log_file.unlink()
                print(f"日志文件 {log_file.name} 已删除")
            except Exception as e:
                print(f"无法删除日志文件 {log_file.name}: {e}")
    else:
        print("日志目录不存在")
                
if __name__ == "__main__":
    setup_logger()