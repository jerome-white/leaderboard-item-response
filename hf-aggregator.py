import sys
import csv
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Queue

from datasets import (
    load_dataset,
    DownloadConfig,
    get_dataset_config_names,
)
from huggingface_hub import HfApi

from logutils import Logger

#
#
#
@dataclass
class DataSetKey:
    key: str
    value: datetime

    def __str__(self):
        return self.key

    def __lt__(self, other):
        return self.value < other.value

@dataclass
class LeaderboardDataset:
    path: Path
    author: str
    model: str

    def __str__(self):
        return str(self.path)

    @classmethod
    def from_path(cls, path):
        (lhs, rhs) = map(path.name.find, ('_', '__'))
        (_lhs, _rhs) = (lhs, rhs)

        if lhs < 0:
            raise ValueError(f'Cannot parse name {path}')

        if lhs == rhs:
            rhs += 2
        elif rhs < 0:
            lhs += 1
        else:
            lhs += 1
            rhs += 2

        if lhs == _lhs:
            author = None
        elif rhs < 0:
            author = path.name[lhs:]
        else:
            author = path.name[lhs:_rhs]
        model = None if rhs == _rhs else path.name[rhs:]

        return cls(path, author, model)

@dataclass
class LeaderboardResult:
    author: str
    model: str
    evaluation: str
    prompt: str
    metric: str
    value: float

#
#
#
def pull(data):
    latest = None
    for i in data.keys():
        try:
            value = datetime.strptime(i, '%Y_%m_%dT%H_%M_%S.%f')
        except ValueError:
            continue
        key = DataSetKey(i, value)
        if latest is None or latest > key:
            latest = key

    if latest is None:
        Logger.warning(f'{data}: keys not timestamped')
        latest = 'latest'
    else:
        latest = str(latest)

    yield from data.get(latest)

def extract(ld_set, evaluation):
    kwargs = asdict(ld_set)
    kwargs.pop('path')

    metrics = (
        'f1',
        'mc2',
        'acc',
        'acc_norm',
    )

    with TemporaryDirectory(suffix=f'.{ld_set.path.name}') as cache_dir:
        data = load_dataset(str(ld_set), evaluation, cache_dir=cache_dir)
        for row in pull(data):
            prompt = row['hashes']['full_prompt']
            for metric in metrics:
                if metric in row:
                    value = float(row[metric])
                    yield LeaderboardResult(
                        evaluation=evaluation,
                        prompt=prompt,
                        metric=metric,
                        value=value,
                        **kwargs,
                    )

def evaluations(ld_set):
    with TemporaryDirectory() as cache_dir:
        dc = DownloadConfig(cache_dir=cache_dir)
        try:
            names = get_dataset_config_names(str(ld_set), download_config=dc)
        except (ValueError, ConnectionError) as err:
            name = type(err).__name__
            msg = f'[{ld_set}] Cannot get config names: {err} ({name})'
            raise ValueError(msg) from err

    for n in names:
        if n.startswith('harness_'):
            yield from extract(ld_set, n)

def func(incoming, outgoing, chunk_size):
    while True:
        ld_set = incoming.get()
        Logger.info(ld_set)

        results = []
        try:
            for i in evaluations(ld_set):
                results.append(asdict(i))
                if len(results) > chunk_size:
                    outgoing.put(results)
                    results = []
        except ValueError as err:
            Logger.error(err)
        finally:
            outgoing.put(results)

        outgoing.put(None)

def ls(author):
    api = HfApi()
    search = 'details_'

    for i in api.list_datasets(author=author, search=search):
        path = Path(i.id)
        assert path.name.startswith(search)
        yield LeaderboardDataset.from_path(path)

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--chunk-size', type=int, default=int(1e5))
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args.chunk_size,
    )

    with Pool(args.workers, func, initargs):
        jobs = 0
        for i in ls(args.author):
            outgoing.put(i)
            jobs += 1

        writer = None
        while jobs:
            rows = incoming.get()
            if rows is None:
                jobs -= 1
            elif rows:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerows(rows)
