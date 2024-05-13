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
    AuthorModel,
    FileChecksum,
    CreationDate,
    EvaluationTask,
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
def walk(info, suffix):
    target = info.with_suffix(suffix)
    if target.exists():
        checksum = FileChecksum(target)
        if checksum:
            raise PermissionError(info)

    with info.open() as fp:
        for i in fp:
            yield Path(i.strip())

def func(queue, args):
    root = Path('datasets', 'open-llm-leaderboard')
    suffix = '.csv.gz'
    dtypes = (
        AuthorModel,
        CreationDate,
        EvaluationTask,
    )
    target = DatasetPathHandler()
    reader = DatasetReader(Backoff(args.backoff, 0.1))

    while True:
        info = queue.get()
        Logger.critical(info.with_suffix(''))

        frames = []
        try:
            for i in walk(info, suffix):
                Logger.info(i)

                rel = i.relative_to(root)
                header = dict(it.chain.from_iterable(
                    asdict(x(y)).items() for (x, y) in zip(dtypes, rel.parts)
                ))

                source = target.to_string(i)
                records = reader.read(source, header)
                df = pd.DataFrame.from_records(records)
                frames.append(df)
        except Exception as err:
            Logger.error('%s (%s)', err, type(err).__name__)
            frames.clear()

        if frames:
            destination = info.with_suffix(suffix)
            df = pd.concat(frames)
            df.to_csv(destination, index=False, compression='gzip')
            checksum = FileChecksum(destination)
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
            queue.put(i)
        queue.join()
