import sys
import uuid
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd

from mylib import Logger

def submission(x):
    return (x
            .groupby(['author', 'model'], sort=False)
            .ngroup())

def func(args):
    (path, target) = args
    Logger.info(path)

    (*_, name, category) = path.parts
    fname = uuid.uuid4()
    dst = (target
           .joinpath(name, category, str(fname))
           .with_suffix('.csv.gz'))
    dst.parent.mkdir()

    df = (pd
          .concat(map(pd.read_csv, path.iterdir()))
          .assign(submission=submission)
    df.to_csv(dst, index=False, compression='gzip')

def each(args):
    for i in args.source.iterdir():
        for j in i.iterdir():
            yield (j, args.target)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--target', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(func, each(args)):
            pass
