import glob

import pandas as pd
import yaml

root_to_dataset_dict = {
    "./data/ENZYMES": "Enzymes",
    "./data/DD": "DD",
    "./data/PROTEINS": "PROTEINS",
    "./data/MUTAG": "MUTAG",
    "./data/Letter-low": "Letter-Low",
}


def tu_root_to_dataset(root):
    return root_to_dataset_dict[root]


# |%%--%%| <OHkyHkANpO|HA4aESCY4C>


results_path = "./results/**/**.yaml"
files = glob.glob(results_path)

# Load all the results into a dataframe.
results = []
for file in files:
    with open(file) as f:
        result = yaml.safe_load(f)
    results.append(result)

df = pd.json_normalize(results)


#######################################################################
### Preprocess the results
#######################################################################

df = df.assign(dataset=df["dataset.root"].apply(tu_root_to_dataset))[
    ["dataset", "model.module", "result.accuracy"]
]

# The count is included to ensure that we can check that no older runs are
# included.
agg_funcs = ["mean", "std", "count"]
df = (
    df.groupby(by=["model.module", "dataset"])
    .agg(agg_funcs)
    .droplevel(level=0, axis=1)
    .unstack()
    .reorder_levels(order=[1, 0], axis=1)
    .sort_index(axis=1)
)

df = df.assign(
    rank_mean=df.xs("mean", level=1, axis=1, drop_level=False)
    .rank(axis=0, ascending=False)
    .mean(axis=1)
)

df

# |%%--%%| <HA4aESCY4C|307efQ4xXU>

# Convert to a latex table

# Slice the AUROC table from it.
df_list = []
for date, new_df in df.round(decimals=2).T.groupby(level=[0]):

    cols_n = list(n[1] for n in new_df.T.columns.to_list())
    if "mean" in cols_n:
        final_df = (
            new_df.T.xs("mean", level=-1, axis=1).astype(str)
            + r" \pm "
            + new_df.T.xs("std", level=-1, axis=1).astype(str)
        )
        df_list.append(final_df)
    else:
        col_name = list(n[0] for n in new_df.T.columns.to_list())
        df_list.append(new_df.T)


df_latex = pd.concat(df_list, axis=1)

df_latex.style.to_latex(
    "./tables/table.tex",
    multirow_align="c",
    multicol_align="c",
    siunitx=True,
)
df_latex


# Store the table
# |%%--%%| <307efQ4xXU|h4PggikYKg>

df_latex.to_markdown(
    "./tables/table.md",
    # multirow_align="c",
    # multicol_align="c",
    # siunitx=True,
)
