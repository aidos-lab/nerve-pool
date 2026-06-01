%load_ext autoreload
%autoreload 2

import networkx as nx 
import torch
from torch_geometric.utils import to_networkx 
from src.models.poolers.nervepool import nerve_pool_complex 
from torch_geometric.data import Data 
import networkx as nx 
import torch
from torch_geometric.utils import to_networkx 
# from src.models.poolers.nervepool import nerve_pool_mesh
from torch_geometric.data import Data 
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

torch.manual_seed(10)

# from datasets.mantra2d import get_dataloaders, DataConfig
#
# config = DataConfig(
#     module="",
#     root="./data/mantra2d",
#     seed=2025,
#     batch_size=1,
#     use_node_attr=False,
#     cleaned=False,
# )
#
# # get_dataset(config=config, force_reload=True)
# dl_train,dl_val,dl_test = get_dataloaders(config=config,force_reload=False)
#
#
# for batch in dl_train:
#     break


#|%%--%%| <BfPKNN1sfh|BYCtCOalMr>
import itertools 


#|%%--%%| <BYCtCOalMr|88TGqfyTQv>

torch.manual_seed(10)


edge_index = torch.tensor(
    [
        [0, 1, 3, 4, 5,6,4],
        [1, 2, 4, 5, 6,3,6],
    ],
)

batch = torch.tensor([0, 0, 0, 0, 0, 0, 0])
input_network = Data(edge_index=edge_index, num_nodes=7)
G = to_networkx(input_network)
pos = nx.spring_layout(G,seed=2)
x = torch.vstack([torch.tensor(p,dtype=torch.float32) for _,p in pos.items()])


edge_features = torch.rand(size=(edge_index.shape[1], x.shape[1]))
face_features = torch.rand(size=(face_index.shape[1],x.shape[1]))
num_virtual_nodes = 5

# g = torch.Generator()
# g.manual_seed(2147483647)
# rand_idx = torch.vstack(
#     [
#         torch.arange(x.shape[0]),
#         torch.randint(0, num_virtual_nodes - 1, size=(x.shape[0], 1),generator = g).squeeze(),
#     ]
# )
# cluster_assignments = torch.zeros(size=(x.shape[0], num_virtual_nodes))
# cluster_assignments[rand_idx[0, :], rand_idx[1, :]] = 1

cluster_assignments = torch.tensor([
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 1., 0.]])

node_color=torch.argmax(cluster_assignments,dim=-1)
print(node_color)
# nx.draw(G,pos=pos, node_color=node_color, cmap=plt.cm.Blues)

# Pool
out_x, out_edge_features, out_edge_index, out_batch_index = nerve_pool_complex(
    node_features=x,
    edge_features=edge_features,
    cluster_assignments=cluster_assignments,
    edge_index=edge_index,
    batch=batch,
)
print(cluster_assignments)
print(x.shape)
print("------")
print(out_edge_index.shape)
print(out_edge_index)
print(out_x.shape)
print(out_x)

print(out_edge_index)

labels_p = { 
          0:"a", 
 1:"b",
 2:"c",
 3:"d"}


out_torch = Data(edge_index=out_edge_index,num_nodes=3)
G_pooled = to_networkx(out_torch)
nx.draw(G_pooled,pos=x_new)
print(out_edge_index)
print(x_new)
print(G_pooled)
