import sys
import csv
import json
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd

from mylib import Logger

#
#
#
class Extractor:
    _vmap = {
        'alpha': 'document',
        'beta': 'document',
        'theta': 'author_model',
    }

    @classmethod
    def select(cls, df):
        for c in df.columns:
            for v in cls._vmap:
                if c.startswith(v):
                    yield c

    def __call__(self, x):
        return x['variable'].apply(self.extract)

    def extract(self, y):
        return self.handle(*y.split('.'))

    def handle(self, name, index):
        raise NotImplementedError()

class ParameterExtractor(Extractor):
    def handle(self, name, index):
        return name

class SourceExtractor(Extractor):
    def __init__(self, db):
        super().__init__()
        self.db = db

    def handle(self, name, index):
        ptype = self._vmap[name]
        return self.db[ptype][index]

#
#
#
def func(incoming, outgoing, args):
    source = SourceExtractor(json.loads(args.parameters.read_text()))
    parameter = ParameterExtractor()
    value_vars = None

    while True:
        (chain, df) = incoming.get()
        Logger.info('%d %d', chain, len(df))
        if value_vars is None:
            value_vars = list(Extractor.select(df))

        records = (df
                   .melt(id_vars='index', value_vars=value_vars)
                   .assign(chain=chain, source=source, parameter=parameter)
                   .drop(columns='variable')
                   .to_dict(orient='records'))
        outgoing.put(records)

def scan(root, window):
    for data in root.iterdir():
        assert data.suffix.endswith('csv')
        (*_, chain) = data.stem.split('_')
        chain = int(chain)
        with pd.read_csv(data, chunksize=window, comment='#') as reader:
            for df in reader:
                yield (chain, df.reset_index())

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--stan-output', type=Path)
    arguments.add_argument('--parameters', type=Path)
    arguments.add_argument('--chunksize', type=int, default=int(1e2))
    # arguments.add_argument('--with-sampler-info', action='store_true')
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
        for i in scan(args.stan_output, args.chunksize):
            outgoing.put(i)
            jobs += 1

        writer = None
        for _ in range(jobs):
            rows = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
