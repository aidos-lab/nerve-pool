"""
Implements the following datasets.

- ENZYMES
- D&D
- REDDIT-MULTI-12K
- COLLAB
- PROTEINS
"""

import os
from dataclasses import dataclass

import torch
import yaml
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.transforms import Compose, ToDense

max_node_dict = {
    "DD": 500,
    "ENZYMES": None,
    "MUTAG": 150,
    "PROTEINS": 700,
    "COLLAB": None,
    "REDDIT-MULTI-12K": None,
    "Letter-low": None,
}

transforms_dict = {
    "DD": None,
    "ENZYMES": None,
    "MUTAG": None,
    "PROTEINS": None,
    "COLLAB": None,
    "REDDIT-MULTI-12K": None,
    "Letter-low": None,
}

prefilter_dict = {
    "DD": lambda data: data.num_nodes <= 500,
    "ENZYMES": None,  # No filter
    "PROTEINS": lambda data: data.num_nodes <= 700,
    "MUTAG": lambda data: data.num_nodes <= 150,
    "NCI1": lambda data: data.num_nodes <= 150,
    "NCI109": lambda data: data.num_nodes <= 150,
    "COLLAB": lambda data: data.num_nodes <= 500,
    "IMDB-BINARY": lambda data: data.num_nodes <= 500,
    "REDDIT-MULTI-12K": lambda data: data.num_nodes <= 700,
    "Letter-low": None,
}


@dataclass
class DataConfig:
    module: str
    batch_size: int
    root: str
    seed: int
    cleaned: bool
    use_node_attr: bool


def get_dataset(config: DataConfig, force_reload: bool = True):
    name = config.root.split("/")[-1]
    transform = transforms_dict[name]
    prefilter = prefilter_dict[name]

    ###########################################
    ### Set up root path
    ###########################################

    os.makedirs(config.root, exist_ok=True)
    ###########################################
    ###########################################

    ds = TUDataset(
        pre_transform=transform,
        pre_filter=prefilter,
        name=name,
        root=config.root,
        cleaned=config.cleaned,
        use_node_attr=config.use_node_attr,
        force_reload=force_reload,
    )

    ###########################################
    ### Save datacard for config validation.
    ###########################################
    data_properties = {
        "in_channels": ds.num_features,
        "num_classes": ds.num_classes,
        "max_num_nodes": max_node_dict[name],
    }
    with open(f"{config.root}/datacard.yaml", "w", encoding="utf-8") as stream:
        # Load dict
        yaml.dump(
            data_properties,
            default_flow_style=False,
            stream=stream,
        )

    return ds


def get_dataloaders(config: DataConfig, force_reload: bool = True):
    """
    Function to create the dataset and return the dataloaders.
    We set the force_reload always to true by default to prevent caching
    issues during development.
    """

    ds = get_dataset(config, force_reload=force_reload)

    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds, test_ds = random_split(
        ds, [0.7, 0.1, 0.2], generator=generator
    )  # type: ignore

    train_dataloader = DataLoader(
        train_ds,  # type: ignore
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_ds,  # type: ignore
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_ds,  # type: ignore
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    """
    Example dataset for testing and as a documentation example.
    As the datasets are rather small, there is no `dev` flag.
    """
    print("hello")
    config = DataConfig(
        module="",
        root="./data/PROTEINS",
        seed=2025,
        batch_size=32,
        use_node_attr=False,
        cleaned=False,
    )

    get_dataloaders(config=config, force_reload=True)
