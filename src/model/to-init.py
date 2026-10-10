"""Computes one shared, data-informed starting point for all chains
(#60), instead of cmdstan's default independent random draw per
chain. alpha is left at its prior mean (1) - the rotational
ambiguity this targets is fundamentally about theta/beta's joint
configuration, not alpha's scale. beta/theta are both simple,
closed-form functions of per-item/per-person accuracy; no fitting
involved.
"""

import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

def init_values(df):
    eps = 1e-3

    item = df.groupby('document_id')['score'].mean().sort_index()
    item = item.clip(eps, 1 - eps)
    beta = np.log((1 - item) / item)  # logit(1 - accuracy) == -logit(accuracy)

    person = df.groupby('author_model_id')['score'].mean().sort_index()
    theta = (person - person.mean()) / person.std()

    return {
        'alpha': [1.0] * len(item),
        'beta': beta.to_list(),
        # Persons 1-2 are fixed anchors (theta[1]=0, theta[2]=1 in
        # model.stan), not part of theta_free.
        'theta_free': theta.iloc[2:].to_list(),
    }

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data-file', type=Path)
    args = arguments.parse_args()

    df = pd.read_csv(args.data_file, memory_map=True)

    print(json.dumps(init_values(df)))
