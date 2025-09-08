from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool


@dataclass
class ModelConfig:
    module: str
    in_channels: int
    hidden_channels: int
    num_classes: int
    hidden_linear_layer: int
    pool_ratio: float
    max_num_nodes: int


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conv1 = GCNConv(config.in_channels, config.hidden_channels)
        self.pool1 = SAGPooling(config.hidden_channels, ratio=config.pool_ratio)
        self.conv2 = GCNConv(config.hidden_channels, config.hidden_channels)
        self.pool2 = SAGPooling(config.hidden_channels, ratio=config.pool_ratio)
        self.conv3 = GCNConv(config.hidden_channels, config.hidden_channels)
        self.lin1 = torch.nn.Linear(config.hidden_channels, config.hidden_linear_layer)
        self.lin2 = torch.nn.Linear(config.hidden_linear_layer, config.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch=batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), 0
