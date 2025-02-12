import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import Pool, Queue

import numpy as np
import pandas as pd

from mylib import Logger

@dataclass
class ItemGroup:
    source: str
    df: pd.DataFrame

    def __str__(self):
        return self.source

class ItemResponseCurve:
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, x):
        with np.errstate(over='ignore'):
            return 1 / (1 + np.exp(-x['alpha'] * (self.theta - x['beta'])))

class ItemIterator:
    _parameter = 'parameter'
    _items = {
        'alpha': 'discrimination',
        'beta':	'difficulty',
    }

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        df = pd.read_csv(self.path, memory_map=True)
        mask = df[self._parameter].isin(self._items)
        index = [
            'chain',
            'sample',
        ]

        for (s, g) in df[mask].groupby('source', sort=False):
            view = (g
                    .pivot(index=index,
                           columns=self._parameter,
                           values='value')
                    .assign(item=s))
            yield ItemGroup(s, view)

def func(incoming, outgoing, args):
    _abilities = np.linspace(
        args.min_ability,
        args.max_ability,
        num=args.n_ability,
    )
    items = ItemIterator._items.items()

    while True:
        group = incoming.get()
        Logger.info(group)

        for a in _abilities:
            irc = ItemResponseCurve(a)
            kwargs = { y: group.df[x].median() for (x, y) in items }
            records = (group
                       .df
                       .assign(irc=irc, ability=a, **kwargs)
                       .replace(to_replace={'irc': {np.inf:  np.nan}})
                       .dropna(subset='irc')
                       .drop(columns=ItemIterator._items)
                       .to_dict(orient='records'))
            outgoing.put(records)
        outgoing.put(None)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--samples', type=Path)
    arguments.add_argument('--min-ability', type=int)
    arguments.add_argument('--max-ability', type=int)
    arguments.add_argument('--n-ability', type=int, default=100)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args,
    )

    with Pool(args.workers, func, initargs):
        jobs = 0
        items = ItemIterator(args.samples)
        for i in items:
            outgoing.put(i)
            jobs += 1

        writer = None
        while jobs:
            rows = incoming.get()
            if rows is None:
                jobs -= 1
            else:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerows(rows)
