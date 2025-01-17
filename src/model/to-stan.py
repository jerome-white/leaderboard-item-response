import json
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd

from mylib import Logger

def func(path):
    Logger.info(path)

    df = pd.read_csv(path, compression='gzip')

    (i, j) = (df[x] for x in ('doc', 'submission'))
    stan = {
        'I': i.max(),                 # questions
        'J': j.max(),                 # persons
        'N': len(df),                 # observations
        'q_i': i.to_list(),           # question for n
        'p_j': j.to_list(),           # person for n
        'y': df['score'].astype(int), # correctness for n
    }

    out = path.with_suffix('.json')
    with out.open('w') as fp:
        print(json.dumps(stan, indent=2), file=fp)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        iterable = args.source.rglob('*.csv')
        for _ in pool.imap_unordered(func, iterable):
            pass
