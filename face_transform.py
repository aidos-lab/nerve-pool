from itertools import combinations

import torch
from mantra.datasets import ManifoldTriangulations
from mantra.representations.one_skeleton import OneSkeleton
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
            OneSkeleton(),
            ManifoldToFace(),
        ]
    ),
)

#|%%--%%| <HelGsI1P3n|FoGn1Iq0Fh>
import numpy as np


def _propagate_values_original(triangulation):
    """Slimmed down version of original."""

    simplices = set([tuple(s) for s in triangulation])
    max_dim = len(next(iter(simplices)))

    for simplex in triangulation:
        for dim in range(1, max_dim):
            simplices.update(s for s in combinations(simplex, r=dim))

    # To sort lexicographically, we need to turn this back into
    # something mutable.
    simplices = list(simplices)
    simplices.sort()
    simplices.sort(key=len)

    # For correctness checks.
    edge_indices = []
    face_indices = []
    for dim in range(1, max_dim):
        simplices_ = [s for s in simplices if len(s) == dim + 1]
        M = []

        for s in simplices_:
            # print(s)
            # View as an array to correct for the index shift; our
            # triangulation is not zero-indexed.
            s = np.asarray(s)

            # Calculate barycenter for the current simplex (i.e., one
            # row of the result matrix).
            edge_tuple = s - 1
            if len(edge_tuple) == 2:
                edge_indices.append(edge_tuple)
            if len(edge_tuple) == 3:
                face_indices.append(edge_tuple)

    return torch.from_numpy(np.vstack(edge_indices).T), torch.from_numpy(np.vstack(face_indices).T)

# |%%--%%| <FoGn1Iq0Fh|IZ30nvGcEf>

data = ds[4]
tri = torch.tensor(data.triangulation).T - 1 
ei, fi = _propagate_values_original(data.triangulation)
assert torch.equal(fi,data.face)
assert torch.equal(tri,data.face)

# Edge [0,4] is not a part of the triangulation and hence wrong.
# Thus the OneSkeleton Transform is not correctly implemented.
print(data.face.T)
print(data.edge_index.T)


