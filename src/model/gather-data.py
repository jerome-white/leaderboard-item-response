from pathlib import Path
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, fields, astuple
from multiprocessing import Pool, JoinableQueue

import pandas as pd

from mylib import Logger

@dataclass
class GroupKey:
    name: str
    category: str

    def __post_init__(self):
        if pd.isnull(self.category):
            self.category = '_'

    def __str__(self):
        return str(self.to_path())

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
    items = [
        'author',
        'model',
        'document',
        'score',
    ]
    reader = DataIterator()

    while True:
        path = queue.get()
        Logger.info(path)

        df = pd.read_csv(path, compression='gzip')
        for (k, g) in reader(df):
            metrics = g['metric']
            if metrics.nunique() > 1:
                Logger.error(
                    '%s -> multiple scoring: %s',
                    k,
                    ' ,'.join(map(str, metrics.unique())),
                )
                continue

            view = g.filter(items=items)

            out = args.target.joinpath(key.to_path())
            out.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode='w', dir=out, delete=False) as fp:
                view.to_csv(fp, index=False)

        queue.task_done()

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
