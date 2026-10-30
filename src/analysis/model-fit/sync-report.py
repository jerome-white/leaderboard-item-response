import sys
import json
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import pandas as pd

from mylib import Logger

class DataReader:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, fname: str):
        path = self.path.joinpath(fname).with_suffix('.csv')
        return pd.read_csv(path, index_col='parameter')

def func(path: Path) -> dict:
    reader = DataReader(path)
    (diagnostics, chain) = map(reader, ('diagnostics', 'chain-sync'))

    stan = path.joinpath('stan').with_suffix('.json')
    spec = json.loads(stan.read_text())

    values = chain.values
    n = values.shape[0]
    mask = ~np.eye(n, dtype=bool)
    offdiag = values[mask]

    selected = diagnostics.loc[chain.index]
    (r_hat, ess_bulk) = (selected[x] for x in ('r_hat', 'ess_bulk'))
    bad_pct = r_hat.gt(1.1).mean() * 100

    return {
        'subject': path.name,
        'items': spec['I'],
        'respondents': spec['J'],
        'sync_mean': offdiag.mean(),
        'sync_min': offdiag.min(),
        'sync_max': offdiag.max(),
        'bad_pct': bad_pct,
        'ess_bulk_min': ess_bulk.min(),
        'ess_bulk_median': ess_bulk.median(),
        'rhat_max': r_hat.max(),
    }

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--results', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        iterable = filter(lambda x: x.is_dir(), args.results.iterdir())
        records = pool.imap_unordered(func, iterable)
        df = (pd
              .DataFrame
              .from_records(records)
              .sort_values('sync_mean', ascending=False))
        df.to_csv(sys.stdout)
