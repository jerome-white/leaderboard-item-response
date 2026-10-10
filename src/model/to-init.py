#
# Compute a shared data-informed starting point for all chains.
#

import json
import functools as ft
from pathlib import Path
from argparse import ArgumentParser

import scipy.special as sp
import numpy as np
import pandas as pd

class MyEncoder(json.JSONEncoder):
    @ft.singledispatchmethod
    def default(self, o):
        return super().default(o)

    @default.register
    def _(self, o: np.ndarray):
        return o.tolist()

class ItemExtractor:
    def __init__(self, df: pd.DataFrame, epsilon: float):
        self.df = df
        self.lower = epsilon
        self.upper = 1 - self.lower

    def __call__(self, column: str) -> np.ndarray:
        return (self
                .df
                .groupby(column)['score']
                .mean()
                .sort_index()
                .clip(self.lower, self.upper)
                .to_numpy())

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data-file', type=Path)
    arguments.add_argument('--epsilon', type=float, default=1e-3)
    args = arguments.parse_args()

    df = pd.read_csv(args.data_file, memory_map=True)

    extract = ItemExtractor(df, args.epsilon)
    (item, person) = map(extract, ('document_id', 'author_model_id'))

    alpha = np.ones(len(item))
    beta = sp.logit(1 - item)
    theta = sp.logit(person)

    # Stan arrays are 1-indexed, so person 1/2 here are Stan's
    # theta[1]/theta[2]. model.stan hardcodes those two as constants
    # (0 and 1), not sampled parameters - theta_free only covers
    # person 3 onward, so that's all we provide init values for here.
    theta = theta[2:]

    data = {
        'alpha': alpha,
        'beta': beta,
        'theta_free': theta,
    }

    print(json.dumps(data, cls=MyEncoder))
