import itertools

import torch
from torch_geometric.utils import to_dense_batch


def nerve_pool_mesh(
    node_features,
    edge_features,
    edge_index,
    face_features,
    face_index,
    cluster_assignments,
    batch,
):
    """
    Performs the pooling for meshes.
    Since a batch of simplicial complexes can be viewed as a larger disconnected graph,
    the down function interleaves with the indexing of torch_geometric. This allows us to
    get away with ignoring the batching alltogether. The same holds for the right function.

    NOTE: It is critical to check that the face indices (and higher order simplices get the proper
    indices). For meshes this is still implemented (e.g. print `batch.face` to check this), but for
    higher order simplices this is no longer automatic.
    """

    device = node_features.device
    num_virtual_nodes = cluster_assignments.shape[1]
    batch_size = batch.max() + 1

    # Initialize _all_ virtual edges, (n choose 2)
    virtual_edge_index = torch.tensor(
        list(
            itertools.combinations(
                range(num_virtual_nodes),
                2,
            )
        ),
        device=device,
    ).T

    # Initialize _all_ virtual faces, (n choose 3)
    virtual_face_index = torch.tensor(
        list(
            itertools.combinations(
                range(num_virtual_nodes),
                3,
            )
        ),
        device=device,
    ).T

    s = cluster_assignments

    # Down Function
    s_edges_virtual_nodes = cluster_assignments[edge_index].max(dim=0)[0]
    s_faces_virtual_nodes = cluster_assignments[face_index].max(dim=0)[0]
    s = torch.vstack([s, s_edges_virtual_nodes, s_faces_virtual_nodes])

    # Right function (naming convention as per the paper.)
    s_all_virtual_edges = s[:, virtual_edge_index].min(dim=1)[0]
    s_all_virtual_faces = s[:, virtual_face_index].min(dim=1)[0]

    num_virtual_edges = s_all_virtual_edges.shape[1]
    num_virtual_faces = s_all_virtual_faces.shape[1]

    # Creates the full cluster assignment matrix (Figure 3 left top).
    # With the batch above this is an exact match.
    s = torch.hstack([s, s_all_virtual_edges, s_all_virtual_faces])

    # Print and compare with the paper.
    # print(s)

    # Prep cluster assignment matrix for batch matrix multiplication.
    batched_edge_index = batch[edge_index][0]
    batched_face_index = batch[face_index][0]
    full_batch_index = torch.hstack(
        [
            batch,
            batched_edge_index,
            batched_face_index,
        ]
    )
    ind = full_batch_index.argsort(dim=0)

    # Converts the S matrix to a batched matrix to convert multiply with the
    # feature matrices.
    dense_s, _ = to_dense_batch(s[ind], full_batch_index[ind])

    # Checks which colums are full zero in the dense matrix.
    # Since removing them now is not possible, we create a mask and postpone this
    # operation to the end.
    non_zero_mask = dense_s.max(dim=1)[0].bool()

    # Features to dense batch as well.
    dense_features, _ = to_dense_batch(
        torch.vstack([node_features, edge_features, face_features])[ind],
        full_batch_index[ind],
    )

    # Batch matrix multiply to obtain the feature assignments.
    pooled_features = torch.bmm(dense_s.movedim(-1, -2), dense_features)

    # Compute the masks for the edges and faces to extract the non-zero rows.
    m_edges = non_zero_mask[
        :, num_virtual_nodes : num_virtual_nodes + num_virtual_edges
    ].reshape(-1)

    m_faces = non_zero_mask[:, num_virtual_nodes + num_virtual_edges :].reshape(-1)

    # Create a batch of sparse edge indices and faces to apply the mask to.
    # Needs to be batched in order to ensure it complies with the torch_geometric
    # indexing conventions for batches.
    sparse_virtual_edge_index = virtual_edge_index.repeat(
        1, batch_size
    ) + num_virtual_nodes * torch.arange(batch_size, device=device).repeat_interleave(
        virtual_edge_index.shape[1]
    ).unsqueeze(0)
    sparse_virtual_face_index = virtual_face_index.repeat(
        1, batch_size
    ) + num_virtual_edges * torch.arange(batch_size, device=device).repeat_interleave(
        virtual_face_index.shape[1]
    ).unsqueeze(0)

    # Return the pooled complex batch.
    return (
        pooled_features[:, :num_virtual_nodes, :].reshape(-1, node_features.shape[1]),
        pooled_features[
            :, num_virtual_nodes : num_virtual_nodes + num_virtual_edges, :
        ].reshape(-1, edge_features.shape[1]),
        pooled_features[
            :,
            num_virtual_nodes + num_virtual_edges : num_virtual_faces
            + num_virtual_nodes
            + num_virtual_edges,
            :,
        ].reshape(-1, edge_features.shape[1]),
        sparse_virtual_edge_index[:, m_edges],
        sparse_virtual_face_index[:, m_faces],
        torch.repeat_interleave(
            torch.arange(batch_size, device=device), num_virtual_nodes
        ),
    )


def nerve_pool_complex(
    node_features,
    edge_features,
    edge_index,
    cluster_assignments,
    batch,
):
    """Pooling for graphs."""
    device = node_features.device
    num_virtual_nodes = cluster_assignments.shape[1]
    batch_size = batch.max() + 1

    # Initialize
    virtual_edge_index = torch.tensor(
        list(
            itertools.combinations(
                range(num_virtual_nodes),
                2,
            )
        ),
        device=device,
    ).T

    s = cluster_assignments

    # Down Function
    s_edges_virtual_nodes = cluster_assignments[edge_index].max(dim=0)[0]
    s = torch.vstack([s, s_edges_virtual_nodes])

    # Right function (naming convention by)
    s_all_virtual_edges = s[:, virtual_edge_index].min(dim=1)[0]

    # Check which columns have all zeros per batch.
    batched_edge_index = batch[edge_index][0]
    # non_empty_mask = s_all_virtual_edges.abs().sum(dim=0).bool()
    # s_all_virtual_edges = s_all_virtual_edges[:, non_empty_mask]

    s = torch.hstack([s, s_all_virtual_edges])
    full_batch_index = torch.hstack([batch, batched_edge_index])
    ind = full_batch_index.argsort(dim=0)

    dense_s, _ = to_dense_batch(s[ind], full_batch_index[ind])
    non_zero_mask = dense_s.max(dim=1)[0].bool()
    dense_features, _ = to_dense_batch(
        torch.vstack([node_features, edge_features])[ind], full_batch_index[ind]
    )

    pooled_features = torch.bmm(dense_s.movedim(-1, -2), dense_features)

    m = non_zero_mask[:, num_virtual_nodes:].reshape(-1)

    sparse_virtual_edge_index = virtual_edge_index.repeat(
        1, batch_size
    ) + num_virtual_nodes * torch.arange(batch_size, device=device).repeat_interleave(
        virtual_edge_index.shape[1]
    ).unsqueeze(0)

    return (
        pooled_features[:, :num_virtual_nodes, :].reshape(-1, node_features.shape[1]),
        pooled_features[:, num_virtual_nodes:, :].reshape(-1, node_features.shape[1]),
        sparse_virtual_edge_index[:, m],
        torch.repeat_interleave(
            torch.arange(batch_size, device=device), num_virtual_nodes
        ),
    )
