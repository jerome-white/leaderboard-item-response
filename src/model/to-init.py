#
# Compute a shared data-informed starting point for all chains.
#

import json
import functools as ft
from pathlib import Path
from argparse import ArgumentParser

import scipy.stats as stats
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

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data-file', type=Path)
    arguments.add_argument('--epsilon', type=float, default=1e-3)
    args = arguments.parse_args()

    df = pd.read_csv(args.data_file, memory_map=True)

    extract = lambda x: df.groupby(x)['score'].mean().sort_index()
    (item, person) = map(extract, ('document_id', 'author_model_id'))
    item = item.clip(args.epsilon, 1 - args.epsilon)

    alpha = np.ones(len(item))
    beta = sp.logit(1 - item)
    theta = stats.zscore(person, ddof=1)

    data = {
        'alpha': alpha,
        'beta': beta,
        'theta_free': theta[2:],
    }

    print(json.dumps(data, cls=MyEncoder))
