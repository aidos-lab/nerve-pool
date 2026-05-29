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
from torch_geometric.data import Data
import yaml
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeRandomTransform
from mantra.representations.one_skeleton import OneSkeleton 
from torchvision.transforms import Compose 

NAME_TO_CLASS_2M = {
        '#^3 T^2': 0, 
        'S^2': 1, 
        '#^4 RP^2': 2, 
        '#^15 RP^2': 3, 
        '#^2 T^2': 4, 
        '#^10 RP^2': 5, 
        '#^5 T^2': 6, 
        '#^6 T^2': 7, 
        'Klein bottle': 8, 
        'T^2': 9, 
        '#^8 RP^2': 10, 
        '#^12 RP^2': 11, 
        '#^7 RP^2': 12, 
        '#^16 RP^2': 13, 
        '#^8 T^2': 14, 
        '#^17 RP^2': 15, 
        'RP^2': 16, 
        '#^5 RP^2': 17, 
        '#^4 T^2': 18, 
        '#^6 RP^2': 19, 
        '#^3 RP^2': 20}


class NameToClass2MTransform:
    """
    Encode the homemorphism type (`name`) as a nominal target for 2-manifolds.
    """

    def __init__(self):
        self.class_dict = NAME_TO_CLASS_2M

    def __call__(self, data: Data):
        assert "name" in data
        data.y = torch.tensor(self.class_dict[data.name]).unsqueeze(0)
        return data

max_node_dict = {
    "mantra2d": None,
}

transforms_dict = {
    "mantra2d": None,
}

prefilter_dict = {
    "mantra2d": None,
}

class ManifoldToFace:
    def __call__(self,data):

        data.face = torch.tensor(data.triangulation).T - 1  
        return data

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

    ds = ManifoldTriangulations(
        root="./data",      # root folder for storing data
        dimension=2,        # Whether to load 2- or 3-manifolds
        version="latest",    # Which version of the dataset to load
        transform=Compose([OneSkeleton(),ManifoldToFace(),NodeRandomTransform(dim=8),NameToClass2MTransform()])
    )

    ###########################################
    ### Save datacard for config validation.
    ###########################################
    data_properties = {
        "in_channels": ds.num_features,
        "num_classes": 21,
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
    config = DataConfig(
        module="",
        root="./data/mantra2d",
        seed=2025,
        batch_size=32,
        use_node_attr=False,
        cleaned=False,
    )

    # get_dataset(config=config, force_reload=True)
    get_dataloaders(config=config,force_reload=False)
