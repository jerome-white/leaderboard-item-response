import sys
import warnings
import functools as ft
from pathlib import Path
from argparse import ArgumentParser

import arviz as az
import pandas as pd
import cmdstanpy

@ft.singledispatch
def scanf(chains: str) -> pd.DataFrame:
    posterior = cmdstanpy.from_csv(path=chains)
    idata = az.from_cmdstanpy(posterior=posterior)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='invalid value encountered in scalar divide',
            category=RuntimeWarning,
        )
        df = az.summary(idata, var_names=['alpha', 'beta', 'theta'])

    return df.query('sd > 0') # drop fixed anchors

@scanf.register
def _(chains: Path) -> pd.DataFrame:
    return scanf(str(chains))

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--chains', type=Path)
    arguments.add_argument('--summary', action='store_true')
    arguments.add_argument('--rhat-threshold', type=float, default=1.1)
    args = arguments.parse_args()

    df = scanf(args.chains)
    if args.summary:
        df = (df
              .assign(family=lambda x: x.index.str.split('[').str[0])
              .groupby('family')
              .agg(n=('r_hat', 'size'),
                   bad=('r_hat', lambda x: x.gt(args.rhat_threshold).sum()),
                   ess_bulk_min=('ess_bulk', 'min'),
                   ess_bulk_median=('ess_bulk', 'median'))
              .assign(bad_pct=lambda x: x['bad'].mul(100).div(x['n'])))
    df.to_csv(sys.stdout)
