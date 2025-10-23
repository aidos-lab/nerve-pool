import os
import pickle
import random
from math import ceil
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from magni.pooling.kmis.kmis_pool import KMISPooling
from magni.pooling.rnd_sparse import RndSparse
from magni.scripts.sum_pool import sum_pool
from magni.scripts.utils import batched_negative_edges
from torch.nn import Linear
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import (
    MLP,
    ASAPooling,
    DenseGINConv,
    DMoNPooling,
    EdgePooling,
    GINConv,
    PANConv,
    PANPooling,
    SAGPooling,
    TopKPooling,
    dense_diff_pool,
    dense_mincut_pool,
    global_add_pool,
    graclus,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import degree, to_dense_adj, to_dense_batch, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes


class EXPWL1Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(EXPWL1Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["EXPWL1.pkl"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/EXPWL1.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
