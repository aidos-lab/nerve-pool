from dataclasses import dataclass

from lightning.fabric.loggers.csv_logs import CSVLogger  # noqa: F401
from lightning.fabric.loggers.tensorboard import TensorBoardLogger


@dataclass
class LogConfig:
    log_dir: str = "base_model"
    model_str: str = "base_model"
    log_interval: int = 10
    logger_name: str = "csvlogger"


def get_logger(config: LogConfig):
    if config.logger_name == "tensorboard":
        logger = TensorBoardLogger(
            f"./logs/{config.model_str}",
            name=config.model_str,
        )
    elif config.logger_name == "csvlogger":
        logger = CSVLogger(
            f"./logs/{config.model_str}",
            name=config.model_str,
            flush_logs_every_n_steps=1,
        )
    else:
        raise ValueError("Provide logger name")

    return logger
