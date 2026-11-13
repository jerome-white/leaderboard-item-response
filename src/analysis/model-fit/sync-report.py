import sys
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import pandas as pd

from mylib import Logger

def read_csv(path: Path, index_col: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col) if path.exists() else None

# Only `diagnostics.csv` is required - every analysis run produces it.
# `chain-sync.csv` and `item-info.csv` are each optional, and each
# contributes its own columns only when present; this function never
# assumes one or the other was (or wasn't) computed over a worst-k
# subset, or computed at all - that's entirely up to whichever stages
# ran upstream.
def func(path: Path) -> dict:
    diagnostics = pd.read_csv(path.joinpath('diagnostics.csv'), index_col='parameter')

    record = {
        'subject': path.name,
        # Always derivable from diagnostics.csv alone, regardless of
        # what else ran: I is the beta count, J is the theta count
        # plus the 2 fixed anchors diagnostics.py drops (sd == 0).
        'items': diagnostics.index.str.startswith('beta[').sum(),
        'respondents': diagnostics.index.str.startswith('theta[').sum() + 2,
    }

    chain = read_csv(path.joinpath('chain-sync.csv'), 'parameter')
    if chain is not None:
        values = chain.values
        n = values.shape[0]
        mask = ~np.eye(n, dtype=bool)
        offdiag = values[mask]
        record.update({
            'sync_mean': offdiag.mean(),
            'sync_min': offdiag.min(),
            'sync_max': offdiag.max(),
        })

    (r_hat, ess_bulk) = (diagnostics[x] for x in ('r_hat', 'ess_bulk'))
    record.update({
        'bad_pct': r_hat.gt(1.1).mean() * 100,
        'ess_bulk_min': ess_bulk.min(),
        'ess_bulk_median': ess_bulk.median(),
        'rhat_max': r_hat.max(),
    })

    item_info = read_csv(path.joinpath('item-info.csv'), 'item')
    if item_info is not None:
        # Does an item's own floor/ceiling-ness (low rate*(1-rate),
        # i.e. low information) predict its beta's R-hat?
        beta = diagnostics[diagnostics.index.str.startswith('beta[')].copy()
        beta['item'] = beta.index.str.extract(r'\[(\d+)\]')[0].astype(int)
        beta = beta.join(item_info, on='item')
        record.update({
            'info_median': item_info['info'].median(),
            'info_rhat_corr': beta['info'].corr(beta['r_hat'], method='spearman'),
        })

    return record

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
              .sort_values('subject'))
        df.to_csv(sys.stdout, index=False)
