from itertools import combinations

import numpy as np
import torch
from mantra.datasets import ManifoldTriangulations
from mantra.representations.one_skeleton import OneSkeleton
from mantra.transforms import MomentCurveEmbedding as MCE_mantra
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torchvision.transforms import Compose

"""Example of what goes wrong."""


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
            MCE_mantra(propagate=True),
            OneSkeleton(),
            ManifoldToFace(),
        ]
    ),
)

# |%%--%%| <HelGsI1P3n|IZ30nvGcEf>


"""
If you read the source code of the moment curve embedding there is no guarantee that edge indices 
of the mce coincide with the ones calculated by the once skeleton. Thus edges and edge features 
DO NOT coincide. 
Moreover, in torch geometric the convention is that an undirected graph is represented with two directed 
edges. 
This is directed:
[
    [0],
    [1],
]
This is undirected:
[
    [0,1],
    [1,0],
]

Thus the mce does:
1) Not even contain the required number of features: (only half as if it were a directed graph). 
2) The edges feature index (index of edge_index.shape[1]) does not coincide with the index of the 
moment curve embedding (index of edge_features.shape[0]). 
3) Due to returning a dict, batching is no longer possible. 
"""

data = ds[0]
print(data)
print("Triangulation:", data.triangulation)
print("Face indices")
print(data.face)
print("Edge indices")
print(data.edge_index)

print("Moment Curve embedding for the nodes:")
# They are not constant. Thus correct edge indices are indeed needed.
print(data.moment_curve_embedding[0])

print("Moment Curve embedding for the edges:")
print(data.moment_curve_embedding[1])
print(data.edge_index.shape)
# assert data.edge_index.shape[1] == data.moment_curve_embedding[1].shape[0]
print("?????")


print("---Face---")
print(data.moment_curve_embedding[2].shape)
print(data.face.shape)

print("Idem for the faces")
# assert data.face.shape[1] == data.moment_curve_embedding[2].shape[0]

# |%%--%%| <IZ30nvGcEf|1cfydFk7Sb>

"""Correct implementation."""

# |%%--%%| <1cfydFk7Sb|5WHu1YL20S>


"""
NOTE: This assigns the moment curve to the nodes randomly.
Mathematically speaking this is not a well-defined function as a reordering of the 
nodes would lead to a different assignment but we'll roll with it for now.
"""


def _calculate_moment_curve(n, d):
    """Calculate moment curve for `n` vertices of a `d`-dimensional manifold.

    This is an auxiliary function for calculating the moment curve of
    `n` vertices of a `d`-dimensional manifold. Notice that the curve
    will have coordinates of dimension `2d + 1`.

    The moment curve is a canonical representation of a triangulation
    but its coordinates are by necessity high-dimensional, and merely
    making use of the number of vertices and the dimension. This will
    mean that the coordinates, by themselves, are not enough to fully
    characterize a triangulation (which is a good thing).

    Parameters
    ----------
    n : int
        Number of vertices

    d : int
        Dimension of the manifold

    Returns
    -------
    np.array of shape (n, 2 * d + 1)
        Coordinates of vertices on the moment curve. Coordinates are
        float values.

    Notes
    -----
    This function is implemented inspired by an article of Francesco
    Mezzadri [1]_.

    References
    ----------
    .. [1] Francesco Mezzadri, "How to Generate Random Matrices from the
    Classical Compact Groups," Notices of the American Mathematical
    Society, Vol. 54, pp. 592--604, 2007.
    """
    t = np.arange(n, dtype=float)
    t /= n - 1

    X = np.vstack([t**k for k in range(1, 2 * d + 2)]).T

    return X


X = _calculate_moment_curve(4, 2)
print(X)


# |%%--%%| <5WHu1YL20S|E5bAl0LvFV>

"""Pass the edge indices as well to calculate it correctly per edge."""


def _propagate_values_original(X, triangulation):
    """Propagate vertex-based values to all simplices.

    This helper function propagates vertex-based values to all simplices
    by calculating the respective barycenter. That is, given any simplex
    of dimension > 0, we will calculate the barycenter of its respective
    values at the vertices.

    Parameters
    ----------
    X : np.array of shape (n, d)
        Vertex-based attributes

    triangulation : list of lists
        A triangulation, expressed as a list of top-level simplices,
        which themselves are lists (or iterable; we do not actually
        care here).

    edge_index: torch.Tensor shape (2,2*num_edges)
        List of undirected edges in the simplicial complex.
        Undirected implies it contains twice the number.
        No assumption on the ordering.

    Returns
    -------
    dict of np.array of shape (n_k, d)
        A dictionary whose keys indicate the respective zero-indexed
        dimension and whose values are the respective values for all
        simplices of that dimension (ordered lexicographically).
    """

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

    values = {
        0: torch.from_numpy(X).to(torch.float32),
    }

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
            M.append(np.mean(X[edge_tuple, :], axis=0))

        M = np.asarray(M)
        values[dim] = torch.from_numpy(M).to(torch.float32)

    return values, np.vstack(edge_indices).T, np.vstack(face_indices).T


# |%%--%%| <E5bAl0LvFV|krAA36IAWE>


def _propagate_values(X, edge_index, face_index):
    """Correct implementation."""
    edge_features = torch.tensor(X, dtype=torch.float32)[edge_index].mean(dim=0)
    face_features = torch.tensor(X, dtype=torch.float32)[face_index].mean(dim=0)
    return edge_features, face_features


# |%%--%%| <krAA36IAWE|hRkzv9eWc0>

# The returned indices from the propagation function and the
# indices from the one skeleton do not coincide.
# This is either a mistake in the propagation function or in the computation of
# the one skeleton.
# Example:
idx = 2
data = ds[idx]
data = ds[idx]
X = _calculate_moment_curve(data.num_nodes, 2)

mce, ei_original, fi_orig = _propagate_values_original(X, data.triangulation)
print(ei_original)
print(data.edge_index)

# |%%--%%| <hRkzv9eWc0|5m5YiEqF0l>

"""
Not finished but it works for the first 2.

Due to the fact that the mce assigns different indices to different 
nodes, compared to the one skeleton transform, it is not possible to 
test the correctness. 
"""

for idx in range(50):
    print(idx)

    data = ds[idx]

    X = _calculate_moment_curve(data.num_nodes, 2)
    ef_correct, ff = _propagate_values(
        X,
        data.edge_index,
        data.face,
    )

    mce, ei_original, fi_orig = _propagate_values_original(X, data.triangulation)
    mce = torch.tensor(mce[1], dtype=torch.float32)
    print("---Edges---")
    print("ORIGINAL")
    print(torch.tensor(ei_original))
    print("DATA")
    print(data.edge_index)
    print("---Faces---")
    print(data.face[:, :6])
    print(torch.tensor(fi_orig)[:, :6])
    # assert torch.equal(torch.tensor(fi_orig), data.face)

    # # Assert that given the indices, they indeed coincide.
    # # Lex sort
    # ind = np.lexsort((data.edge_index[1].tolist(), data.edge_index[0].tolist()))
    #
    # # Sort with lex
    # ef_correct = ef_correct[ind]
    # ei_correct = data.edge_index.clone()[:, ind]
    #
    # # print(torch.round(torch.tensor(mce), decimals=2))
    #
    # # assert torch.equal(torch.tensor(ei_original,dtype=torch.int32),ei_correct)
    # #
    # # Mask to remove the reverse edges. The edge_indices are sorted lexicographically.
    # # mask = ei_correct[0] < ei_correct[1]
    # # print(ei_correct[:,mask])
    # # print(ei_original)
    #
    # # # # print(ef_correct[mask])
    # # # # print(mce[1])
    # # # print(data.edge_index[:,mask])
    # # print(ef_correct[mask].shape)
    # # print(mce)
    # #
    # # # Test passes, thus the impl is correct.
    # # assert torch.allclose(ef_correct[mask], mce)
