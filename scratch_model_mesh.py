%load_ext autoreload
%autoreload 2

from dataclasses import dataclass
import networkx as nx 
import torch
from torch_geometric.utils import to_networkx 
from src.models.poolers.nervepool import nerve_pool_mesh
from torch_geometric.data import Data 
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

torch.manual_seed(10)

from datasets.mantra2d import get_dataloaders, DataConfig

config = DataConfig(
    module="",
    root="./data/mantra2d",
    seed=2025,
   batch_size=1,
    use_node_attr=False,
    cleaned=False,
)

# get_dataset(config=config, force_reload=True)
dl_train,dl_val,dl_test = get_dataloaders(config=config,force_reload=False)


for batch in dl_train:
    break


#|%%--%%| <88TGqfyTQv|BYCtCOalMr>

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
    def __init__(self, in_channels, config: ModelConfig,num_virt_nodes: int=5):
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
                num_virt_nodes,
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
        self.gnn2_embed = GCNConv(config.hidden_channels, 5)
        self.gnn2_pool = GNN(config.hidden_channels, config)

        # Third layer
        self.gnn3_embed = GCNConv(config.hidden_channels, config.hidden_channels)

        # Final Aggragation
        self.lin1 = torch.nn.Linear(config.hidden_channels, config.hidden_linear_layer)
        self.lin2 = torch.nn.Linear(config.hidden_linear_layer, config.num_classes)
        self.config = config

    def forward(self, data):

        x, edge_index,face_index = data.x, data.edge_index, data.face

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

        face_features = torch.zeros(
            size=(face_index.shape[1], x.shape[1]), device=x.device
        )

        print("x",x.shape)
        print("edge_features",edge_features.shape,edge_index.shape)
        print("s",s.shape)
        # Try argmax.
        # Try to add edge features.
        # Pool.
        x, edge_features, face_features, edge_index,face_index, batch_index = nerve_pool_mesh(
            node_features=x,
            cluster_assignments=s,
            edge_features=edge_features,
            edge_index=edge_index,
            face_features=face_features,
            face_index=face_index,
            batch=data.batch,
        )
        print("x",x.shape)
        print("edge_features",edge_features.shape,edge_index.shape)

        # Second layer
        s = self.gnn2_pool(x, edge_index)
        x = self.gnn2_embed(x, edge_index)
        x = F.relu(x)

        print(edge_index)
        print(face_index.shape)
        print(x)
        print(batch_index.shape)

        # # Loss
        # adj = to_dense_adj(edge_index, batch_index)
        # s_dense, _ = to_dense_batch(s, batch=batch_index)
        #
        # link_loss = adj - torch.matmul(s_dense, s_dense.transpose(1, 2))
        # link_loss = torch.norm(link_loss, p=2)
        # link_loss_2 = link_loss / adj.numel()
        #
        # ent_loss_2 = (-s_dense * torch.log(s_dense + 1e-15)).sum(dim=-1).mean()
        #
        # # breakpoint()
        # # link_loss_2 = F.mse_loss(torch.bmm(s_dense, s_dense.movedim(-1, -2)), adj)
        #
        # # Pool
        # x, edge_features, face_features, edge_index,face_index, batch_index = nerve_pool_mesh(
        #     node_features=x,
        #     cluster_assignments=s,
        #     edge_features=edge_features,
        #     edge_index=edge_index,
        #     face_features=face_features,
        #     face_index=face_index,
        #     batch=data.batch,
        # )
        #
        # # Third layer
        # x = self.gnn3_embed(x, edge_index)
        # x = F.relu(x)
        #
        # # Potential bug
        # # x = x.mean(dim=1)
        # x = global_mean_pool(x, batch_index)
        # x = self.lin1(x)
        # x = F.relu(x)
        # x = self.lin2(x)
        # return F.log_softmax( x, dim=-1), link_loss_1 + link_loss_2 + ent_loss_1 + ent_loss_2

import copy

config = ModelConfig(
    module="",
  hidden_channels= 64,
  hidden_linear_layer= 64,
  in_channels= 8,
  max_num_nodes= 500,
  normalize= False,
  num_classes= 2,
  use_linear_layer= True)

model = Model(config)


model(copy.deepcopy(batch))

#|%%--%%| <BYCtCOalMr|n5kjWluqHm>



#|%%--%%| <n5kjWluqHm|D1nZcTtZR3>



#|%%--%%| <D1nZcTtZR3|OTI84azVTr>


# Pool
x_new, edge_features, face_features, out_edge_index,out_face_index, out_batch_index = nerve_pool_mesh(
    node_features=x,
    cluster_assignments=cluster_assignments,
    edge_features=edge_features,
    edge_index=edge_index,
    face_features=face_features,
    face_index=face_index,
    batch=batch,
)
print(cluster_assignments)
print(x.shape)
print("------")
print(out_edge_index.shape)
print(out_edge_index)
print(out_face_index)
print(x_new.shape)
print(x_new)
labels_p = { 
          0:"a", 
 1:"b",
 2:"c",
 3:"d"}
 
print(out_edge_index)
out_torch = Data(edge_index=out_edge_index,num_nodes=4)
G_pooled = to_networkx(out_torch)
nx.draw(G_pooled,pos=x_new,labels=labels_p)



