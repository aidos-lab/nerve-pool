"""Makes all datasets"""

from datasets.tu import DataConfig, get_dataloaders


def main():
    roots = [
        "./data/DD",
        "./data/MUTAG",
        "./data/PROTEINS",
        "./data/ENZYMES",
        "./data/COLLAB",
        "./data/REDDIT-MULTI-12K",
    ]
    for root in roots:
        print(root)
        config = DataConfig(
            module="",
            root=root,
            seed=2025,
            batch_size=32,
            use_node_attr=False,
            cleaned=False,
        )
        _ = get_dataloaders(config, force_reload=True)
