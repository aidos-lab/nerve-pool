import itertools

import torch
from torch_geometric.utils import to_dense_batch


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
