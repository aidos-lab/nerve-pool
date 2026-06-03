from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from models.poolers.nervepool import nerve_pool_complex, nerve_pool_mesh


@dataclass
class ModelConfig:
    module: str
    in_channels: int
    num_classes: int
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
        self.lin = torch.nn.Linear(
            config.hidden_channels,
            3,
        )

    def bn(self, i, x):
        x = getattr(self, f"bn{i}")(x)
        return x

    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())
        x = self.lin(x)
        return F.softmax(x, dim=-1)


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        ##############################################
        ### First layer
        ##############################################

        # Pooling model
        self.gnn1_pool_edges = GNN(config.in_channels, config)

        # Embedding
        self.gnn1_embed_nodes = GCNConv(config.in_channels, config.hidden_channels)
        self.gnn1_embed_edges = nn.Linear(config.in_channels, config.hidden_channels)
        self.gnn1_embed_faces = nn.Linear(config.in_channels, config.hidden_channels)

        ##############################################
        ### Second layer
        ##############################################

        # Pooling model
        self.gnn2_pool_edges = GNN(config.hidden_channels, config)

        # Embedding
        self.gnn2_embed_nodes = GCNConv(config.hidden_channels, config.hidden_channels)
        self.gnn2_embed_edges = nn.Linear(
            config.hidden_channels, config.hidden_channels
        )
        self.gnn2_embed_faces = nn.Linear(
            config.hidden_channels, config.hidden_channels
        )

        ##############################################
        ### Third layer
        ##############################################

        # Embedding
        self.gnn3_embed_nodes = GCNConv(config.hidden_channels, config.hidden_channels)
        self.gnn3_embed_edges = nn.Linear(
            config.hidden_channels, config.hidden_channels
        )
        self.gnn3_embed_faces = nn.Linear(
            config.hidden_channels, config.hidden_channels
        )

        ##############################################
        ### Final Aggragation
        ##############################################

        self.lin1 = torch.nn.Linear(config.hidden_channels, config.hidden_linear_layer)
        self.lin2 = torch.nn.Linear(config.hidden_linear_layer, config.num_classes)

    def forward(self, data):
        (
            x,
            edge_index,
            edge_features,
            face_index,
            face_features,
        ) = (
            data.x,
            data.edge_index,
            data.edge_features,
            data.face,
            data.face_features,
        )

        # First layer
        s = self.gnn1_pool_edges(x, edge_index)
        x = self.gnn1_embed_nodes(x, edge_index)
        x = F.relu(x)
        edge_features = self.gnn1_embed_edges(edge_features)
        face_features = self.gnn1_embed_faces(face_features)

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

        # Pool.
        x, edge_features, face_features, edge_index, face_index, batch_index = (
            nerve_pool_mesh(
                node_features=x,
                edge_features=edge_features,
                face_features=face_features,
                face_index=face_index,
                cluster_assignments=s,
                edge_index=edge_index,
                batch=data.batch,
            )
        )

        ##############################################
        ### Second layer
        ##############################################

        s = self.gnn2_pool_edges(x, edge_index)
        x = self.gnn2_embed_nodes(x, edge_index)
        x = F.relu(x)

        edge_features = self.gnn2_embed_edges(edge_features)
        face_features = self.gnn2_embed_faces(face_features)

        adj = to_dense_adj(
            edge_index,
            batch_index,
            max_num_nodes=self.config.max_num_nodes,
        )
        s_dense, _ = to_dense_batch(
            s,
            batch=batch_index,
            max_num_nodes=self.config.max_num_nodes,
        )

        link_loss = adj - torch.matmul(s_dense, s_dense.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        link_loss_1 = link_loss / adj.numel()

        ent_loss_1 = (-s_dense * torch.log(s_dense + 1e-15)).sum(dim=-1).mean()

        # Pool.
        x, edge_features, face_features, edge_index, face_index, batch_index = (
            nerve_pool_mesh(
                node_features=x,
                edge_features=edge_features,
                face_features=face_features,
                face_index=face_index,
                cluster_assignments=s,
                edge_index=edge_index,
                batch=batch_index,
            )
        )

        # Second layer
        x = self.gnn3_embed_nodes(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), link_loss_1 + ent_loss_1
