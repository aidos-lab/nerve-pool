from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


@dataclass
class ModelConfig:
    module: str
    in_channels: int
    hidden_channels: int
    hidden_linear_layer: int
    num_classes: int
    max_num_nodes: int
    use_linear_layer: bool
    normalize: bool


class GNN(torch.nn.Module):
    def __init__(self, in_channels, config: ModelConfig):
        super().__init__()
        self.conv1 = DenseGCNConv(in_channels, config.hidden_channels, config.normalize)
        self.bn1 = torch.nn.BatchNorm1d(config.hidden_channels)
        self.conv2 = DenseGCNConv(
            config.hidden_channels, config.hidden_channels, config.normalize
        )
        self.bn2 = torch.nn.BatchNorm1d(config.hidden_channels)
        self.conv3 = DenseGCNConv(
            config.hidden_channels, config.hidden_channels, config.normalize
        )
        self.bn3 = torch.nn.BatchNorm1d(config.hidden_channels)
        if config.use_linear_layer:
            self.lin = torch.nn.Linear(config.hidden_channels, config.hidden_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f"bn{i}")(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())
        if self.lin is not None:
            x = self.lin(x).relu()
        return x


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # First layer
        self.gnn1_embed = DenseGCNConv(config.in_channels, config.hidden_channels)

        # Not the cleanest solution.
        self.gnn1_pool = GNN(config.in_channels, config)
        # Second layer
        self.gnn2_embed = DenseGCNConv(config.hidden_channels, config.hidden_channels)
        # Not the cleanest solution.
        self.gnn2_pool = GNN(config.hidden_channels, config)

        # Third layer
        self.gnn3_embed = DenseGCNConv(config.hidden_channels, config.hidden_channels)

        # Final Aggragation
        self.lin1 = torch.nn.Linear(config.hidden_channels, config.hidden_linear_layer)
        self.lin2 = torch.nn.Linear(config.hidden_linear_layer, config.num_classes)

    def forward(self, data):
        x, mask = to_dense_batch(
            data.x, data.batch, max_num_nodes=self.config.max_num_nodes
        )

        adj = to_dense_adj(
            data.edge_index, data.batch, max_num_nodes=self.config.max_num_nodes
        )

        # First layer
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)

        # Pool
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        # Second layer
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)

        # Pool
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        # Third layer
        x = self.gnn3_embed(x, adj)
        x = F.relu(x)

        # Potential bug
        x = x.mean(dim=1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), e1 + l1 + e2 + l2
