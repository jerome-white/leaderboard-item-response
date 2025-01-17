import json
import functools as ft
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd

from mylib import Logger

class MyEncoder(json.JSONEncoder):
    @ft.singledispatchmethod
    def default(self, obj):
        return super().default(obj)

    @default.register
    def _(self, obj: pd.Series):
        return obj.to_list()

def func(path):
    Logger.info(path)

    df = pd.read_csv(path)
    score = df['score']
    if not score.apply(float.is_integer).all():
        Logger.error(path)
        return

    (i, j) = (df[x] for x in ('document', 'author_model_id'))
    stan = {
        'I': i.max(),           # questions
        'J': j.max(),           # persons
        'N': len(df),           # observations
        'q_i': i,               # question for n
        'p_j': j,               # person for n
        'y': score.astype(int), # correctness for n
    }

    out = path.with_suffix('.json')
    with out.open('w') as fp:
        print(json.dumps(stan, indent=2, cls=MyEncoder), file=fp)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        iterable = args.source.rglob('*.csv')
        for _ in pool.imap_unordered(func, iterable):
            pass
