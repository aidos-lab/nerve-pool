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
    device = node_features.device
    num_virtual_nodes = cluster_assignments.shape[1]
    batch_size = batch.max() + 1
    print("batch_size", batch_size)


    ########################################################
    ## Edges
    ########################################################


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

    num_virtual_edges = virtual_edge_index.shape[1]

    # Initialize
    virtual_face_index = torch.tensor(
        list(
            itertools.combinations(
                range(num_virtual_nodes),
                3,
            )
        ),
        device=device,
    ).T
    print(virtual_face_index)

    num_virtual_faces = virtual_face_index.shape[1]
    s = cluster_assignments

    # Down Function
    s_edges_virtual_nodes = cluster_assignments[edge_index].max(dim=0)[0]
    s_faces_virtual_nodes = cluster_assignments[face_index].max(dim=0)[0]
    s = torch.vstack([s, s_edges_virtual_nodes,s_faces_virtual_nodes])
    

    # Right function (naming convention by)
    s_all_virtual_edges = s[:, virtual_edge_index].min(dim=1)[0]

    # Check which columns have all zeros per batch.
    batched_edge_index = batch[edge_index][0]
    # non_empty_mask = s_all_virtual_edges.abs().sum(dim=0).bool()
    # s_all_virtual_edges = s_all_virtual_edges[:, non_empty_mask]

    s = torch.hstack([s, s_all_virtual_edges])
    full_batch_index = torch.hstack([batch, batched_edge_index])
    # ind = full_batch_index.argsort(dim=0)
    #
    # dense_s, _ = to_dense_batch(s[ind], full_batch_index[ind])
    # non_zero_mask = dense_s.max(dim=1)[0].bool()
    # dense_features, _ = to_dense_batch(
    #     torch.vstack([node_features, edge_features])[ind], full_batch_index[ind]
    # )
    #
    # pooled_features = torch.bmm(dense_s.movedim(-1, -2), dense_features)
    #
    # m = non_zero_mask[:, num_virtual_nodes:].reshape(-1)
    #
    # sparse_virtual_edge_index = virtual_edge_index.repeat(
    #     1, batch_size
    # ) + num_virtual_nodes * torch.arange(batch_size, device=device).repeat_interleave(
    #     virtual_edge_index.shape[1]
    # ).unsqueeze(
    #     0
    # )


    ########################################################
    ## Faces
    ########################################################

    # Right function (naming convention by)
    s_all_virtual_faces = s[:, virtual_face_index].min(dim=1)[0]

    # Check which columns have all zeros per batch.
    batched_face_index = batch[face_index][0]
    # non_empty_mask = s_all_virtual_edges.abs().sum(dim=0).bool()
    # s_all_virtual_edges = s_all_virtual_edges[:, non_empty_mask]

    s = torch.hstack([s, s_all_virtual_faces])
    full_batch_index = torch.hstack([full_batch_index, batched_face_index])
    ind = full_batch_index.argsort(dim=0)

    dense_s, _ = to_dense_batch(s[ind], full_batch_index[ind])
    non_zero_mask = dense_s.max(dim=1)[0].bool()
    dense_features, _ = to_dense_batch(
        torch.vstack([node_features, edge_features,face_features])[ind], full_batch_index[ind]
    )

    pooled_features = torch.bmm(dense_s.movedim(-1, -2), dense_features)

    m_edges = non_zero_mask[:, num_virtual_nodes:num_virtual_nodes+num_virtual_edges].reshape(-1)
    m_faces = non_zero_mask[:, num_virtual_nodes+num_virtual_edges:].reshape(-1)

    sparse_virtual_edge_index = virtual_edge_index.repeat(
        1, batch_size
    ) + num_virtual_nodes * torch.arange(batch_size, device=device).repeat_interleave(
        virtual_edge_index.shape[1]
    ).unsqueeze(
        0
    )
    sparse_virtual_face_index = virtual_face_index.repeat(
        1, batch_size
    ) + num_virtual_faces * torch.arange(batch_size, device=device).repeat_interleave(
        virtual_face_index.shape[1]
    ).unsqueeze(
        0
    )
    print(m_edges.shape)
    print(m_faces.shape)

    return (
        pooled_features[:, :num_virtual_nodes, :].reshape(-1, node_features.shape[1]),
        pooled_features[:, num_virtual_nodes:num_virtual_nodes+num_virtual_edges, :].reshape(-1, edge_features.shape[1]),
        pooled_features[:, num_virtual_nodes+num_virtual_edges:num_virtual_faces + num_virtual_nodes+num_virtual_edges, :].reshape(-1, edge_features.shape[1]),
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
    ).unsqueeze(
        0
    )

    return (
        pooled_features[:, :num_virtual_nodes, :].reshape(-1, node_features.shape[1]),
        pooled_features[:, num_virtual_nodes:, :].reshape(-1, node_features.shape[1]),
        sparse_virtual_edge_index[:, m],
        torch.repeat_interleave(
            torch.arange(batch_size, device=device), num_virtual_nodes
        ),
    )
