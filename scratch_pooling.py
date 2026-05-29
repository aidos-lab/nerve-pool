import networkx as nx 
import torch
from torch_geometric.utils import to_networkx 
from src.models.poolers.nervepool import nerve_pool_complex 
from torch_geometric.data import Data 

#|%%--%%| <pMqWxzM48Z|88TGqfyTQv>

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
x_new, edge_features, out_edge_index, out_batch_index = nerve_pool_complex(
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
print(x_new.shape)
print(x_new)

#|%%--%%| <88TGqfyTQv|OTI84azVTr>
out_torch = Data(x=x,edge_index=out_edge_index)


G = to_networkx(out_torch, node_attrs=x_new.numpy())
nx.draw(G)
