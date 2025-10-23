import argparse
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from loaders import load_config, load_datamodule, load_logger, load_model, save_config

torch.set_float32_matmul_precision("high")
# torch.use_deterministic_algorithms(mode=False)
torch.manual_seed(0)

DEVICE = "cuda"


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


# def train_loop(epoch, model, train_loader, optimizer, logger):
def train_loop(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        # print(i, data.num_nodes)
        data = data.to(DEVICE)
        optimizer.zero_grad()
        output, link_loss = model(data)
        loss = F.nll_loss(output, data.y.view(-1)) + link_loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return Result(loss=total_loss)


@torch.no_grad()
def evaluation_loop(model, loader) -> Result:
    model.eval()
    correct = 0
    loss = 0
    for data in loader:
        data = data.to(DEVICE)
        output, link_loss = model(data)
        loss += F.nll_loss(output, data.y.view(-1)) + link_loss
        pred = output.max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return Result(accuracy=correct / len(loader.dataset), loss=loss.item())


def train_model_from_config(config):
    #####################################################
    ### Load dataset
    #####################################################
    dm = load_datamodule(config.dataset)
    train_loader, val_loader, test_loader = dm.get_dataloaders(
        config.dataset, force_reload=False
    )

    # Model
    model = load_model(config.model).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.trainer.lr)

    # Loggers
    logger = load_logger(config.logger)

    early_stop_patience = 150
    tolerance = 0.0001
    best_val_acc = 0
    epochs_no_improve = 0

    # Train model
    for epoch in range(1, config.trainer.epochs + 1):
        train_metrics = train_loop(model, train_loader, optimizer)
        print("Epoch ", epoch, "Loss: ", train_metrics.loss)

        val_metrics = evaluation_loop(model, val_loader)
        test_metrics = evaluation_loop(model, test_loader)

        logger.log_metrics(
            to_dict(train_metrics, prefix="train")
            | to_dict(val_metrics, prefix="val")
            | to_dict(test_metrics, prefix="test"),
            step=epoch,
        )

        # Early stopping
        if val_metrics.accuracy > best_val_acc + tolerance:
            best_val_acc = val_metrics.accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Run the test set.
    result = evaluation_loop(model, test_loader)
    logger.log_metrics(to_dict(result, prefix="test"), step=-1)
    config.result = result

    print("Test acc", result.accuracy)

    # Save results.
    folder_name = config.experiment.name.split("-")[0]

    # TODO: Needs some improvement.
    os.makedirs(f"./results/{folder_name}", exist_ok=True)
    result_filename = f"./results/{folder_name}/{config.experiment.name}.yaml"

    save_config(
        config,
        result_filename,
    )

    # # Save model
    # torch.save(
    #     model.state_dict(),
    #     "logs/{}/checkpoint_{}.pt".format(config.experiment_, str(epoch)),
    # )


def main():
    """
    Trains a single model from a single configuration.
    Calling the script 'train' from the commandline invokes
    this script.
    """
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
    train_model_from_config(config)


if __name__ == "__main__":
    sys.exit(main())
