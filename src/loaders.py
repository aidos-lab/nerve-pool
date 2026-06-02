"""
Helpers to load a class from a configuration file.
Sectioned into models, datasets, loggers.
"""

import importlib
import json
from types import SimpleNamespace
from typing import Any, Callable

import yaml

"""
Helpers to load a class from a configuration file.
Sectioned into models, datasets, loggers.
"""


def load_module(config: dict[str, str]) -> tuple[Callable[dict[str, str], None], Any]:
    # If the module does not exist, write error. This may happen if configs contain error.
    module = importlib.import_module(config["module"])
    if not hasattr(module, "setup"):
        raise ValueError(
            f"Trying to load {config['module']}, but the setup function is missing."
        )
    config, cls = module.setup()
    return config, cls


def load_context(full_config_dict):
    """Loads the context from the configuration."""
    ctx_dict = {}
    for key, config_dict in full_config_dict.items():
        if "module" not in config_dict.keys():
            raise ValueError(f"Trying to load {key}, but the 'module' key is missing.")

        # If the module is not none, it is a string to a module that needs loading.
        # For configs without a module associated (i.e. such as the training configs,
        # this key is set to None.
        if config_dict["module"] is not None:
            # Load the python module.
            config_cls, module_cls = load_module(config_dict)
            # Create the instance of the Config class (returned by the setup function).
            # The setup function returns a tuple of the form (config_class, setup_fn).
            # The latter takes an instance of the former as input and the config dictionary
            # Can be used to initialize the config class.
            config_instance = config_cls(**config_dict)

            # Write the initialized module to the key in the context dictionary.
            ctx_dict[key] = module_cls(config_instance)
        else:
            # Only create a namesapce with the provided configs.
            # No classes or setup to be done.
            ctx_dict[key] = SimpleNamespace(**config_dict)

    return SimpleNamespace(**ctx_dict)


#######################################################################
### Configuration
#######################################################################


# @timeit_decorator
def load_object(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    else:
        return obj


def load_config(path: str):
    """
    Loads the configuration yaml and parses it into an object with dot access.
    """
    with open(path, encoding="utf-8") as stream:
        # Load dict
        config_dict = yaml.safe_load(stream)

        # Convert to namespace (access via config.data etc)
        config = json.loads(json.dumps(config_dict), object_hook=load_object)
    return config


def save_config(config, path: str):
    """
    Save the configuration yaml.
    """
    print(f"Saving config to {path}")
    with open(path, "w", encoding="utf-8") as stream:
        # Load dict
        yaml.dump(
            json.loads(json.dumps(config, default=lambda s: vars(s))),
            default_flow_style=False,
            stream=stream,
        )


def config_to_dict(config):
    """
    Converts nested namespace to nested dictionary.
    Needed for printing."""
    return json.loads(
        json.dumps(config, default=lambda s: vars(s)),
    )


def print_config(config):
    print(
        yaml.dump(
            config_to_dict(config),
            default_flow_style=False,
        )
    )


#######################################################################
### Datasets
#######################################################################


def load_datamodule(config):
    # Validation
    if not hasattr(config, "module"):
        raise ValueError("Path to the module is missing.")
    return importlib.import_module(config.module)


#######################################################################
### Models
#######################################################################


def load_model(config):
    # Validation
    if not hasattr(config, "module"):
        raise ValueError("Path to the module is missing.")
    module = importlib.import_module(config.module)
    model_class = getattr(module, "Model")
    return model_class(config)


def load_model_config(config):
    # Validation
    if not hasattr(config, "module"):
        raise ValueError("Path to the module is missing.")
    module = importlib.import_module(config.module)
    model_config_class = getattr(module, "ModelConfig")
    return model_config_class


#######################################################################
### Loggers
#######################################################################


def load_logger(config):
    # Validation
    module = importlib.import_module("loggers")
    return module.get_logger(config)


def load_logger_config(_):
    # Validation
    module = importlib.import_module("loggers")
    logger_config_class = getattr(module, "LogConfig")
    return logger_config_class
