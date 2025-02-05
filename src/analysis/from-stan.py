import sys
import csv
import json
from pathlib import Path
from argparse import ArgumentParser

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
    def select(cls, values):
        for c in values:
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
def scan(root, chunksize, sample=None):
    usecols = None
    comment = '#'

    for data in root.iterdir():
        assert data.suffix.endswith('csv')
        (*_, chain) = data.stem.split('_')
        chain = int(chain)

        if usecols is None:
            with data.open() as fp:
                iterable = filter(lambda x: not x.startswith(comment), fp)
                reader = csv.DictReader(iterable)
                usecols = list(Extractor.select(reader.fieldnames))

        with pd.read_csv(data,
                         comment=comment,
                         usecols=usecols,
                         chunksize=chunksize) as reader:
            for df in reader:
                yield (chain, df)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--stan-output', type=Path)
    arguments.add_argument('--parameters', type=Path)
    arguments.add_argument('--read-size', type=int, default=int(1e2))
    arguments.add_argument('--write-size', type=int, default=int(1e4))
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    source = SourceExtractor(json.loads(args.parameters.read_text()))
    parameter = ParameterExtractor()

    writer = None
    for (chain, df) in scan(args.stan_output, args.read_size):
        frame = (df
                 .reset_index()
                 .melt(id_vars='index', value_vars=df.columns)
                 .assign(chain=chain, source=source, parameter=parameter)
                 .drop(columns='variable'))
        n = len(frame)

        Logger.info(
            '%d [%d, %d) -> %d',
            chain,
            df.index.start,
            df.index.stop,
            n,
        )

        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=frame.columns)
            writer.writeheader()

        for i in range(0, n, args.write_size):
            j = i + args.write_size
            view = frame.iloc[i:j]
            if view.empty:
                break
            writer.writerows(view.to_dict(orient='records'))
