import copy

import torch
from mantra.datasets import ManifoldTriangulations
from mantra.representations.internal import SimplexTrie
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx
from torchvision.transforms import Compose


class ManifoldToFace:
    def __call__(self, data):

        data.face = torch.tensor(data.triangulation).T - 1
        return data


ds = ManifoldTriangulations(
    root="./data",  # root folder for storing data
    dimension=2,  # Whether to load 2- or 3-manifolds
    version="latest",  # Which version of the dataset to load
    transform=Compose(
        [
            # NodeRandomTransform(dim=8),
            # NameToClass2MTransform(),
            # MCE_mantra(propagate=True),
            # OneSkeleton(),
            ManifoldToFace(),
        ]
    ),
)


#|%%--%%| <3PWe5E7HOg|xDvb87r9QS>

class OneSkeleton(BaseTransform):
    def forward(self, data):
        G = self._build_one_skeleton(data["triangulation"])
        print(G.edges)
        print(G.nodes)
        data_ = from_networkx(G)
        # NOTE: Node 0,4 is  printed, this is wrong.
        print(data_.edge_index.T)

        for k, v in data_.items():
            assert k not in data
            data[k] = v

        return data

    def _build_one_skeleton(self, top_simplices):
        simplex_trie = SimplexTrie()
        for s in top_simplices:
            simplex_trie.insert(s)

        one_simplices = sorted(
            node.simplex for node in simplex_trie.skeleton(1)
        )

        G = nx.Graph()

        for s in one_simplices:
            u, v = list(s)

            for w in [u, v]:
                if w  not in G:  
                    G.add_node(w)

            # NOTE: Node 1,5 is not printed, as it should.
            print(s,u,v)

            G.add_edge(u , v )  

        return G


# The problem is that node [0,4] / [1,5] is not in the triangulation.

tr = OneSkeleton()
data = ds[4]
data_out = tr(copy.deepcopy(data))
print(data_out.edge_index.T[:10] + 1)
print(torch.tensor(data.triangulation))



#|%%--%%| <xDvb87r9QS|4IDLLZsb48>


class OneSkeletonCorrect(BaseTransform):
    def forward(self, data):
        ei = self._build_one_skeleton(data["triangulation"])
        data.edge_index = ei
        return data

    def _build_one_skeleton(self, top_simplices):
        # First we construct the Trie to optimally extract the 1-skeleton
        simplex_trie = SimplexTrie()
        for s in top_simplices:
            simplex_trie.insert(s)

        one_simplices = sorted(
            node.simplex for node in simplex_trie.skeleton(1)
        )
        ei = []
        for s in one_simplices:
            ei.append(list(s))
        return torch.tensor(ei)-1


tr = OneSkeleton()
data = ds[4]
data_out = tr(copy.deepcopy(data))
print(data_out.edge_index.T[:10] + 1)
print(torch.tensor(data.triangulation))




