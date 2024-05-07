import sys
import csv
import itertools as it
from pathlib import Path
from hashlib import blake2b
from datetime import datetime
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Queue

import pandas as pd

from mylib import Logger, DatasetPathHandler, hf_datetime

@dataclass
class CreationDate:
    date: datetime

    def __post_init__(self):
        self.date = hf_datetime(self.date)

@dataclass
class EvaluationTask:
    task: str
    category: str

    def __init__(self, info):
        (_, name, _) = info.split('|')
        parts = name.split('_', maxsplit=1)
        n = len(parts)
        assert 0 < n <= 2

        self.task = parts[0]
        self.category = parts[1] if n > 1 else ''

@dataclass
class AuthorModel:
    author: str
    model: str

    def __init__(self, name):
        # parse the values
        (lhs, rhs) = map(name.find, ('_', '__'))
        if lhs < 0:
            raise ValueError(f'Cannot parse name {name}')
        (_lhs, _rhs) = (lhs, rhs)

        # calculate the bounds
        if lhs == rhs:
            rhs += 2
        elif rhs < 0:
            lhs += 1
        else:
            lhs += 1
            rhs += 2

        # extract the names
        if lhs == _lhs:
            self.author = None
        elif rhs < 0:
            self.author = name[lhs:]
        else:
            self.author = name[lhs:_rhs]
        self.model = None if rhs == _rhs else name[rhs:]

#
#
#
class DatasetIterator:
    _digest_size = 16

    def __init__(self, df):
        self.df = df

    def __iter__(self):
        for i in self.df.itertuples(index=False):
            instruction = i.full_prompt.encode()
            message = blake2b(instruction, digest_size=self._digest_size)
            prompt = message.hexdigest()

            for (m, v) in self.metrics(i):
                yield {
                    'prompt': prompt,
                    'metric': m,
                    'value': float(v),
		}

    def metrics(self, data):
        raise NotImplementedError()

class NestedDatasetIterator(DatasetIterator):
    def metrics(self, data):
        yield from data.metrics.items()

class ExplicitDatasetIterator(DatasetIterator):
    _prefixes = (
        '',
        'metric.',
    )
    _metrics = (
        'em',
        'f1',
        'mc1',
        'mc2',
        'acc',
        'acc_norm',
    )

    def metrics(self, data):
        data = data._asdict()
        for (i, j) in it.product(self._prefixes, self._metrics):
            key = i + j
            if key in data:
                yield (j, data[key])

#
#
#
def func(incoming, outgoing):
    root = Path('datasets', 'open-llm-leaderboard')
    dtypes = (
        AuthorModel,
        CreationDate,
        EvaluationTask,
    )
    target = DatasetPathHandler()

    while True:
        path = incoming.get()
        Logger.info(path)

        header = {}
        records = []

        rel = path.relative_to(root)
        for (i, j) in zip(dtypes, rel.parts):
            header.update(asdict(i(j)))

        df = pd.read_parquet(target.to_string(path))
        if 'metrics' in df.columns:
            iterator = NestedDatasetIterator
        else:
            iterator = ExplicitDatasetIterator

        for i in iterator(df):
            rec = dict(header)
            rec.update(i)
            records.append(rec)

        outgoing.put(records)
#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    # arguments.add_argument('--max-retries', type=int, default=5)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
    )

    with Pool(args.workers, func, initargs):
        jobs = 0
        for i in sys.stdin:
            outgoing.put(Path(i.strip()))
            jobs += 1

        writer = None
        for _ in range(jobs):
            rows = incoming.get()
            if rows:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerows(rows)
