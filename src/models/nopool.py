from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


@dataclass
class ModelConfig:
    module: str
    in_channels: int
    max_num_nodes: int
    hidden_channels: int
    hidden_linear_layer: int
    num_classes: int


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gnn1_embed = GCNConv(config.in_channels, config.hidden_channels)
        self.gnn2_embed = GCNConv(config.hidden_channels, config.hidden_channels)
        self.gnn3_embed = GCNConv(config.hidden_channels, config.hidden_channels)
        self.lin1 = torch.nn.Linear(config.hidden_channels, config.hidden_linear_layer)
        self.lin2 = torch.nn.Linear(config.hidden_linear_layer, config.num_classes)
        self.config = config

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gnn1_embed(x, edge_index)
        x = F.relu(x)
        x = self.gnn2_embed(x, edge_index)
        x = F.relu(x)
        x = self.gnn3_embed(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch=batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), 0
