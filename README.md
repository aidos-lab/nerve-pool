# Documentation 

Additional notes on the experiments can be found under the notes folder. 


# Installation 

The virtual environment can be installed with the command `uv sync`. 
Ensure that it gets created under `.venv` in the root directory as pyright 
is configured to check for this folder. 

# Datasets 

For our experiments we use the TU datasets. The dataset is downloaded on the fly, however it is usually 
convenient to preprocess the data before training the model. The datasets can be downloaded and preprocessed 
by running the following command in the terminal.  

```shell
uv run datasets
```

# Training 

To train a model using the configuration provided, run the train script with a 
provided configuration which can be found under the config folder. One can train 
a single model with the following command. 

```shell
uv run src/train.py configs/nervepool/nervepool-DD-2020.yaml
```

The `sweep-*.yaml` files are used to generate the set of configs for the experiments 
in each of the subfolders. To create a set of configs with such a sweep file, we run the 
command `uv run sweep path/to/sweep-file.yaml`.

The training script will create a similar named file in the `results` directory and are under 
version control. To create the tables, run the script `tables.py` under the `notebooks` folder.




