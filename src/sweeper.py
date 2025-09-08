"""Generates configs from a sweep."""

import argparse
import copy
import importlib
import os
import sys
from dataclasses import asdict
from itertools import product

import yaml


def load_datacard(config):
    path = f"{config['root']}/datacard.yaml"
    with open(path, encoding="utf-8") as stream:
        # Load dict
        datacard_dict = yaml.safe_load(stream)
    return datacard_dict


def load_model_config(config):
    module = importlib.import_module(config["module"])
    model_config_class = getattr(module, "ModelConfig")
    return asdict(model_config_class(**config))


def _gen_combinations(d):
    keys, values = d.keys(), d.values()

    list_keys = [k for k in keys if isinstance(d[k], list)]
    nonlist_keys = [k for k in keys if k not in list_keys]
    list_values = [v for v in values if isinstance(v, list)]
    nonlist_values = [v for v in values if v not in list_values]

    combinations = product(*list_values)

    for c in combinations:
        result = dict(zip(list_keys, c))
        result.update({k: v for k, v in zip(nonlist_keys, nonlist_values)})
        yield result


def compute_product_dictionary(d):
    """https://stackoverflow.com/questions/50606454/cartesian-product-of-nested-dictionaries-of-lists"""
    keys, values = d.keys(), d.values()

    dict_values = [v for v in values if isinstance(v, dict)]
    dict_keys = [k for k in keys if isinstance(d[k], dict)]
    nondict_values = [v for v in values if v not in dict_values]
    nondict_keys = [k for k in keys if k not in dict_keys]

    for c in product(*(_gen_combinations(v) for v in dict_values)):
        result = dict(zip(dict_keys, c))
        result.update({k: v for k, v in zip(nondict_keys, nondict_values)})
        yield result


def load_sweep(config_dict):

    config_list = []
    dict_list = compute_product_dictionary(config_dict)
    for _config in dict_list:
        config = copy.deepcopy(_config)
        data_card = load_datacard(config["dataset"])
        model_config = load_model_config(config["model"] | data_card)
        config["model"] = model_config
        # Add experiment config
        name = f"{config['logger']['log_dir']}_{config['dataset']['seed']}"
        config["logger"]["log_dir"] = name
        config_list.append(config)

    return config_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input configuration")
    args = parser.parse_args()

    with open(args.INPUT, encoding="utf-8") as stream:
        # Load dict
        config_dict = yaml.safe_load(stream)

    config_folder = f"./configs/{config_dict['model']['module'].split('.')[-1]}"
    os.makedirs(config_folder, exist_ok=True)

    configs = load_sweep(config_dict)

    for config in configs:
        model_name = config["model"]["module"].split(".")[-1]
        dataset_name = config["dataset"]["root"].split("/")[-1]
        seed = config["dataset"]["seed"]

        filename = f"./configs/{model_name}/{model_name}-{dataset_name}-{seed}.yaml"
        config["experiment"] = {"name": f"{model_name}-{dataset_name}-{seed}"}
        with open(filename, "w", encoding="utf-8") as stream:
            # Load dict
            yaml.dump(
                config,
                default_flow_style=False,
                stream=stream,
            )


if __name__ == "__main__":
    sys.exit(main())
