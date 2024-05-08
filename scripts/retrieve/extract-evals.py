import sys
import csv
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
from multiprocessing import Pool, Queue

import pandas as pd
from pydantic import TypeAdapter, ValidationError
from huggingface_hub.utils import HfHubHTTPError

from mylib import Logger, Backoff, DatasetPathHandler, hf_datetime

@ft.cache
def clean(name, delimiter='_'):
    letters = []
    for n in name:
        l = delimiter if n in string.punctuation else n
        letters.append(l)

    return ''.join(letters)

class FloatAdapter:
    def __init__(self):
        (self.to_float, self.to_bool) = map(TypeAdapter, (float, bool))

    def __call__(self, value):
        for i in (self.to_float, self.to_bool):
            try:
                return i(value)
            except ValidationError:
                pass

        raise ValueError(f'Cannot convert {value} to float')

    def to_float(self, value):
        return self.f.validate_python(value)

    def to_bool(self, value):
        return self.to_float(self.b.validate_python(value))

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

    def __init__(self, df, recorder):
        self.df = df
        self.record = recorder
        self.to_float = FloatAdapter()

    def __iter__(self):
        for i in self.df.itertuples(index=False):
            instruction = i.full_prompt.encode()
            message = blake2b(instruction, digest_size=self._digest_size)
            prompt = message.hexdigest()
            self.record(prompt, i.full_prompt)

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
class PromptRecorder:
    def __call__(self, p_id, text):
        raise NotImplementedError()

class NoOpPromptRecorder(PromptRecorder):
    def __call__(self, p_id, text):
        return

class UniquePromptRecorder(PromptRecorder):
    def __init__(self, root, task):
        super().__init__()
        self.output = root.joinpath(task.to_path())

    def __call__(self, p_id, text):
        name = str(uuid.uuid4())
        output = self.output.joinpath(p_id, name)
        try:
            output.parent.mkdir(parents=True)
        except FileExistsError:
            return

        output.write_text(text)

#
#
#
class DatasetReader:
    def __init__(self, backoff):
        self.backoff = backoff

    def read(self, target):
        for (i, j) in enumerate(self.backoff, 1):
            try:
                return pd.read_parquet(target)
            except (HfHubHTTPError, FileNotFoundError) as err:
                Logger.error('%s (attempt=%d, backoff=%ds)', err, i, j)
            time.sleep(j)

def func(incoming, outgoing, args):
    root = Path('datasets', 'open-llm-leaderboard')
    dtypes = ( # do not reorder!
        AuthorModel,
        CreationDate,
        EvaluationTask,
    )
    target = DatasetPathHandler()
    reader = DatasetReader(Backoff(15, 0.1))

    while True:
        path = incoming.get()
        Logger.info(path)

        rel = path.relative_to(root)
        info = [ x(y) for (x, y) in zip(dtypes, rel.parts) ]

        # Establish the header
        header = {}
        for i in info:
            header.update(asdict(i))

        #
        if args.save_prompts:
            recorder = UniquePromptRecorder(args.save_prompts, info[-1])
        else:
            recorder = NoOpPromptRecorder()

        #
        df = reader.read(target.to_string(path))
        if 'metrics' in df.columns:
            iterator = NestedDatasetIterator
        else:
            iterator = ExplicitDatasetIterator

        #
        records = []
        for i in iterator(df, recorder):
            rec = dict(header)
            rec.update(i)
            records.append(rec)
        outgoing.put(records)
#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--save-prompts', type=Path)
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
