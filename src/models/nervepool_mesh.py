from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from models.poolers.nervepool import nerve_pool_complex


@dataclass
class ModelConfig:
    module: str
    in_channels: int
    num_classes: int
    use_linear_layer: bool
    normalize: bool
    hidden_linear_layer: int
    hidden_channels: int
    max_num_nodes: int


class GNN(torch.nn.Module):
    def __init__(self, in_channels, config: ModelConfig):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels,
            config.hidden_channels,
            config.normalize,
        )
        self.bn1 = torch.nn.BatchNorm1d(config.hidden_channels)
        self.conv2 = GCNConv(
            config.hidden_channels,
            config.hidden_channels,
            config.normalize,
        )
        self.bn2 = torch.nn.BatchNorm1d(config.hidden_channels)
        self.conv3 = GCNConv(
            config.hidden_channels,
            config.hidden_channels,
            config.normalize,
        )
        self.bn3 = torch.nn.BatchNorm1d(config.hidden_channels)
        if config.use_linear_layer:
            self.lin = torch.nn.Linear(
                config.hidden_channels,
                config.hidden_channels,
            )
        else:
            self.lin = None

    def bn(self, i, x):
        x = getattr(self, f"bn{i}")(x)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())
        # x = self.conv1(x, adj, mask).relu()
        # x = self.conv2(x, adj, mask).relu()
        # x = self.conv3(x, adj, mask).relu()
        if self.lin is not None:
            x = self.lin(x).relu()
        return F.softmax(x, dim=-1)


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # First layer
        self.gnn1_embed = GCNConv(config.in_channels, config.hidden_channels)
        self.gnn1_pool = GNN(config.in_channels, config)
        # Second layer
        self.gnn2_embed = GCNConv(config.hidden_channels, config.hidden_channels)
        self.gnn2_pool = GNN(config.hidden_channels, config)

        # Third layer
        self.gnn3_embed = GCNConv(config.hidden_channels, config.hidden_channels)

        # Final Aggragation
        self.lin1 = torch.nn.Linear(config.hidden_channels, config.hidden_linear_layer)
        self.lin2 = torch.nn.Linear(config.hidden_linear_layer, config.num_classes)
        self.config = config

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First layer
        s = self.gnn1_pool(x, edge_index)
        x = self.gnn1_embed(x, edge_index)
        x = F.relu(x)
        adj = to_dense_adj(
            edge_index,
            data.batch,
            max_num_nodes=self.config.max_num_nodes,
        )
        s_dense, _ = to_dense_batch(
            s,
            batch=data.batch,
            max_num_nodes=self.config.max_num_nodes,
        )

        link_loss = adj - torch.matmul(s_dense, s_dense.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        link_loss_1 = link_loss / adj.numel()

        ent_loss_1 = (-s_dense * torch.log(s_dense + 1e-15)).sum(dim=-1).mean()

        edge_features = torch.zeros(
            size=(edge_index.shape[1], x.shape[1]), device=x.device
        )

        # Try argmax.
        # Try to add edge features.
        # Pool.
        x, edge_features, edge_index, batch_index = nerve_pool_complex(
            node_features=x,
            edge_features=edge_features,
            cluster_assignments=s,
            edge_index=edge_index,
            batch=data.batch,
        )

        # Second layer
        s = self.gnn2_pool(x, edge_index)
        x = self.gnn2_embed(x, edge_index)
        x = F.relu(x)

        # Loss
        adj = to_dense_adj(edge_index, batch_index)
        s_dense, _ = to_dense_batch(s, batch=batch_index)

        link_loss = adj - torch.matmul(s_dense, s_dense.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        link_loss_2 = link_loss / adj.numel()

        ent_loss_2 = (-s_dense * torch.log(s_dense + 1e-15)).sum(dim=-1).mean()

        # breakpoint()
        # link_loss_2 = F.mse_loss(torch.bmm(s_dense, s_dense.movedim(-1, -2)), adj)

        # Pool
        x, edge_features, edge_index, batch_index = nerve_pool_complex(
            node_features=x,
            edge_features=edge_features,
            cluster_assignments=s,
            edge_index=edge_index,
            batch=batch_index,
        )

        # Third layer
        x = self.gnn3_embed(x, edge_index)
        x = F.relu(x)

        # Potential bug
        # x = x.mean(dim=1)
        x = global_mean_pool(x, batch_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.log_softmax(
            x, dim=-1
        ), link_loss_1 + link_loss_2 + ent_loss_1 + ent_loss_2
