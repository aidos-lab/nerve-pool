"""Analysis of the datasets"""

import glob

import numpy as np
import pandas as pd
import torch

from loaders import load_config, load_datamodule

config_files = glob.glob("./configs/nopool/**-2025.yaml")
configs = [load_config(f).dataset for f in config_files]

data_list = []

for c in configs:
    ds = load_datamodule(c).get_dataset(c, force_reload=False)

    for g in ds:
        data_list.append(
            {
                "name": c.root.split("/")[-1].lower(),
                "num_nodes": g.num_nodes,
                "num_edges": g.num_edges,
                "num_features": g.x.shape[1],
            }
        )


# |%%--%%| <LmoNhRd6T9|HWRvzTBZvp>

df = pd.DataFrame(data_list)

# fmt: off
(
    df
    .groupby("name")
    .agg(["mean", "std", "min", "max", "count"])
)
# fmt: on
# |%%--%%| <HWRvzTBZvp|ILsLpoX9hQ>
