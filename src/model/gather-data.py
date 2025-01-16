import sys
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, fields, astuple
from multiprocessing import Pool, JoinableQueue

import pandas as pd

from mylib import Logger

@dataclass
class GroupKey:
    name: str
    category: str

    def __post_init__(self):
        if not self.category:
            self.category = '_'

    def to_path(self):
        return Path(*astuple(self))

class DataIterator:
    def __init__(self):
        self.by = [ x.name for x in fields(GroupKey) ]

    def __call__(self, df):
        for (i, g) in df.groupby(self.by, sort=False, dropna=False):
            key = GroupKey(*i)
            yield (key, g)

def func(queue, args):
    reader = DataIterator()

    while True:
        path = queue.get()
        Logger.info(path)

        df = pd.read_csv(path, compression='gzip')
        for (k, g) in reader(df):
            args.target.joinpath(k.to_path())

def each(args):
    for i in args.source.rglob('*.csv.gz'):
        yield (i, args.target)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--target', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args,
    )

    with Pool(args.workers, func, initargs):
        for i in args.source.rglob('*.csv.gz'):
            queue.put(i)
        queue.join()
