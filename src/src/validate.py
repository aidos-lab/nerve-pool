import argparse
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from loaders import (
    config_to_dict,
    load_config,
    load_datamodule,
    load_logger,
    load_logger_config,
    load_model,
    load_model_config,
    save_config,
)

torch.set_float32_matmul_precision("high")
# torch.use_deterministic_algorithms(mode=False)
torch.manual_seed(0)


@dataclass
class ExperimentConfig:
    name: str


@dataclass
class TrainConfig:
    epochs: int = 2
    lr: float = 0.001


@dataclass
class Result:
    loss: float = np.nan
    accuracy: float = np.nan


def to_dict(result: Result, prefix: str | None = None):
    res_dict = asdict(result)
    if prefix is not None:
        res_dict = {f"{prefix}/{k}": v for k, v in res_dict.items()}
    return res_dict


def validate_config(config):

    #####################################################
    ### Load dataset
    #####################################################

    print("--- Validating ---")

    print("Validating data configuration")
    dm = load_datamodule(config.dataset)
    # Instantiate the config object
    data_config = dm.DataConfig(**config_to_dict(config.dataset))
    print(data_config)

    print("Validating Model configuration")
    # Model
    model_config_class = load_model_config(config.model)
    model_config = model_config_class(**config_to_dict(config.model))
    print(model_config)

    print("Validating Logging configuration")
    # Loggers
    logger_config_class = load_logger_config(config.logger)
    logger_config = logger_config_class(**config_to_dict(config.logger))
    print(logger_config)

    print("--Validation Succesful--")


def main():
    #####################################################
    ### Argparse
    #####################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input configuration")
    parser.add_argument(
        "--dev", default=False, action="store_true", help="Run a small subset."
    )
    args = parser.parse_args()
    config_path = args.INPUT

    #####################################################
    ### Load configs
    #####################################################
    config = load_config(config_path)

    print(f"Validating file: {config_path}")
    validate_config(config)


if __name__ == "__main__":
    sys.exit(main())
