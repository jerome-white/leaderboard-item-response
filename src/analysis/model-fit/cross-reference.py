import sys
import json
from typing import Iterable
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

def cross(chains: pd.DataFrame, items: pd.DataFrame) -> Iterable[dict]:
    beta = 'beta['
    n = len(beta)

    for parameter in chains.index:
        if parameter.startswith(beta):
            # Same 0-indexed (cmdstanpy/arviz) vs. 1-indexed (q_i,
            # from stan.json) offset as sync-report.py's item/beta
            # join.
            index = parameter[n:-1]
            item = int(index) + 1
            if item in items.index:
                mean_abs_corr = (chains
                                 .loc[parameter]
                                 .drop(parameter)
                                 .abs()
                                 .mean())
                info = items.loc[item, 'info']

                yield {
                    'parameter': parameter,
                    'item': item,
                    'mean_abs_corr': mean_abs_corr,
                    'info': info,
                }

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--subject')
    arguments.add_argument('--chain-sync', type=Path)
    arguments.add_argument('--item-info', type=Path)
    args = arguments.parse_args()

    subject = {
        'subject': args.subject,
    }
    chains = pd.read_csv(args.chain_sync, index_col='parameter')
    items = pd.read_csv(args.item_info, index_col='item')

    for c in cross(chains, items):
        c |= subject
        print(json.dumps(c))
