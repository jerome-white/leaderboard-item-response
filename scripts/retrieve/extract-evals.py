import uuid
import time
import string
import itertools as it
import functools as ft
from pathlib import Path
from hashlib import blake2b
from datetime import datetime
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from multiprocessing import Pool, JoinableQueue

import pandas as pd
from pydantic import TypeAdapter, ValidationError
from huggingface_hub.utils import HfHubHTTPError

from mylib import (
    Logger,
    Backoff,
    FileChecksum,
    DatasetPathHandler,
    hf_datetime,
)

@ft.cache
def clean(name, delimiter='_'):
    letters = []
    for n in name:
        l = delimiter if n in string.punctuation else n
        letters.append(l)

    return ''.join(letters)

class FloatAdapter:
    def __init__(self):
        (self.ftype, self.btype) = map(TypeAdapter, (float, bool))

    def __call__(self, value):
        for i in (self.to_float, self.to_bool):
            try:
                return i(value)
            except ValidationError:
                pass

        raise ValueError(f'Cannot convert {value} to float')

    def to_float(self, value):
        return self.ftype.validate_python(value)

    def to_bool(self, value):
        return self.to_float(self.btype.validate_python(value))

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

    def to_path(self):
        args = map(clean, (self.task, self.category))
        return Path(*args)

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
        self.to_float = FloatAdapter()

    def __iter__(self):
        for i in self.df.itertuples(index=False):
            instruction = i.full_prompt.encode()
            message = blake2b(instruction, digest_size=self._digest_size)
            prompt = message.hexdigest()

            for (m, v) in self.metrics(i):
                try:
                    value = self.to_float(v)
                except ValueError as err:
                    Logger.warning(err)
                    continue

                yield {
                    'prompt': prompt,
                    'metric': m,
                    'value': value,
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
class DatasetReader:
    def __init__(self, backoff):
        self.backoff = backoff

    def get(self, target):
        for (i, j) in enumerate(self.backoff, 1):
            try:
                return pd.read_parquet(target)
            except (HfHubHTTPError, FileNotFoundError) as err:
                Logger.error('%s (attempt=%d, backoff=%ds)', err, i, j)
            time.sleep(j)

        raise LookupError(target)

    def read(self, target, header):
        df = self.get(target)
        if 'metrics' in df.columns:
            iterator = NestedDatasetIterator
        else:
            iterator = ExplicitDatasetIterator

        for i in iterator(df):
            record = dict(header)
            record.update(i)

            yield record

#
#
#
def func(queue, args):
    root = Path('datasets', 'open-llm-leaderboard')
    dtypes = (
        AuthorModel,
        CreationDate,
        EvaluationTask,
    )
    target = DatasetPathHandler()
    reader = DatasetReader(Backoff(args.backoff, 0.1))

    while True:
        info = queue.get()

        path = Path(info.read_text().strip())
        rel = path.relative_to(root)
        iterable = (asdict(x(y)).items() for (x, y) zip(dtypes, rel.parts))
        header = dict(it.chain.from_iterable(iterable))

        # Read/write the data
        src = target.to_string(path)
        dst = info.with_suffix('.csv.gz')

        Logger.info('%s %s', info, path)
        try:
            records = reader.read(src, header, recorder)
            df = pd.DataFrame.from_records(records)
            df.to_csv(dst, index=False, compression='gzip')
        except Exception as err:
            Logger.critical('%s: %s (%s)', path, err, type(err).__name__)

        # Checksum the write
        if dst.exists():
            checksum = FileChecksum(dst)
            checksum.write()

        queue.task_done()

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--index-path', type=Path)
    arguments.add_argument('--backoff', type=float, default=15)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args,
    )

    with Pool(args.workers, func, initargs):
        for i in args.index_path.rglob('*.info'):
            target = i.with_suffix('.csv.gz')
            if target.exists():
                checksum = FileChecksum(target)
                if checksum:
                    Logger.warning(f'Skipping {i}')
                    continue
            queue.put(i)
        queue.join()
