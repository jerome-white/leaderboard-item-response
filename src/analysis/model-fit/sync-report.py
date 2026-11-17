import sys
import json
import functools as ft
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from mylib import Logger

class SyncEncoder(json.JSONEncoder):
    @ft.singledispatchmethod
    def default(self, o):
        return super().default(o)

    @default.register
    def _(self, o: np.int64 | np.float64):
        return o.item()

def parameter(df: pd.DataFrame, name: str) -> pd.Series:
    return df.index.str.startswith(f'{name}[')

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--subject')
    arguments.add_argument('--diagnostics', type=Path)
    arguments.add_argument('--chain-sync', type=Path)
    arguments.add_argument('--item-info', type=Path)
    args = arguments.parse_args()

    diagnostics = pd.read_csv(args.diagnostics, index_col='parameter')
    record = {
        'subject': args.subject,
    }

    (r_hat, ess_bulk) = (diagnostics[x] for x in ('r_hat', 'ess_bulk'))
    record.update({
        'bad_pct': r_hat.gt(1.1).mean() * 100,
        'ess_bulk_min': ess_bulk.min(),
        'ess_bulk_median': ess_bulk.median(),
        'rhat_max': r_hat.max(),
        # Always derivable from diagnostics.csv alone, regardless of
        # what else ran: I is the beta count, J is the theta count
        # plus the 2 fixed anchors diagnostics.py drops (sd == 0).
        'items': parameter(diagnostics, 'beta').sum(),
        'respondents': parameter(diagnostics, 'theta').sum() + 2,
    })

    if args.chain_sync is not None:
         chains = pd.read_csv(args.chain_sync, index_col='parameter')
         values = chains.values
         n = values.shape[0]
         mask = ~np.eye(n, dtype=bool)
         offdiag = values[mask]
         record.update({
             'sync_mean': offdiag.mean(),
             'sync_min': offdiag.min(),
             'sync_max': offdiag.max(),
         })


    # Does an item's own floor/ceiling-ness (low rate*(1-rate),
    # i.e. low information) predict its beta's R-hat?
    if args.item_info is not None:
        items = pd.read_csv(args.item_info, index_col='item')
        beta = diagnostics[parameter(diagnostics, 'beta')].copy()
        params = beta.index.str.extract(r'\[(\d+)\]')

        # cmdstanpy/arviz label array elements 0-indexed (beta[0]..),
        # while item-info.csv's item column is q_i straight out of
        # stan.json, Stan's own 1-indexed item id - off by one.
        beta['item'] = params[0].astype(int).values + 1

        beta = beta.join(items, on='item')
        record.update({
            'info_median': items['info'].median(),
            'info_rhat_corr': beta['info'].corr(
                beta['r_hat'],
                method='spearman',
            ),
        })

    print(json.dumps(record, cls=SyncEncoder))
