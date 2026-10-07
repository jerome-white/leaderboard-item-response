import sys
import csv
import time
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from dataclasses import dataclass, asdict, fields, replace
from multiprocessing import Pool, Queue
from collections.abc import Iterable, Iterator

from datasets import load_dataset
from huggingface_hub import DatasetInfo, HfApi, HfFileSystem

from mylib import (
    Dataset,
    Logger,
    Backoff,
    DatasetPathHandler,
    ModelDatabase,
    ModelMetadata,
    retry_after,
)

#
#
#
class ModelIterator:
    _dtype = '-details'

    def __init__(self, author):
        self.author = author
        self.api = HfApi()

    def __iter__(self):
        yield from self.api.list_datasets(
            author=self.author,
            search=self._dtype,
        )

#
#
#
@dataclass
class DatasetListing:
    row: dict
    dataset: Dataset

class DatasetIterator:
    def __init__(self, author: str):
        dataset = Dataset(author, 'contents')
        self.datasets = load_dataset(str(dataset), split='train')

    def __iter__(self):
        for row in self.datasets:
            dataset = Dataset.from_fullname(row['fullname'])
            yield DatasetListing(row, dataset)

class ModelHandler:
    _dtype = ModelIterator._dtype

    def __init__(self, author: str, handler=None):
        self.author = author
        self.handler = handler

    def __iter__(self):
        handler = self
        while handler is not None:
            yield handler.handle
            handler = handler.handler

    def __call__(self, model: Iterable[DatasetInfo]) -> Iterator[DatasetInfo]:
        for m in model:
            for h in self:
                result = h(m)
                if result is None:
                    break
            else:
                yield m

    def __getitem__(self, item: DatasetInfo) -> Dataset:
        dataset = Dataset.from_leaderboard(item.id, self.author)
        name = dataset.name.removesuffix(self._dtype)
        return replace(dataset, name=name)

    def handle(self, model: DatasetInfo):
        return model

class FlaggedHandler(ModelHandler):
    def __init__(self, handler, datasets: DatasetIterator):
        super().__init__(handler.author, handler)
        iterable = filter(lambda x: x.row['Flagged'], datasets)
        self.flagged = set(x.dataset for x in iterable)

    def handle(self, model: DatasetInfo):
        if self[model] not in self.flagged:
            return model

class DatabaseHandler(ModelHandler):
    def __init__(self, handler, datasets: DatasetIterator, db: ModelDatabase):
        super().__init__(handler.author, handler)
        self.db = db
        self.metadata = { x.dataset: x.row for x in datasets }

    def handle(self, model: DatasetInfo):
        dataset = self[model]
        row = self.metadata.get(dataset)
        if row is not None:
            value = ModelMetadata(
                author=dataset.namespace,
                model=dataset.name,
                mtype=row['Type'],
                precision=row['Precision'],
                params=row['#Params (B)'],
                merged=row['Merged'],
            )
            self.db.put(value)

        return model

#
#
#
@dataclass
class Result:
    path: Path
    date: datetime

    def __repr__(self):
        (*prefix, _) = self.path.stem.split('_')
        return '/'.join(prefix)

    def __lt__(self, other):
        return self.date < other.date

class DatasetFileSystem:
    def __init__(self, backoff):
        self.backoff = backoff
        self.fs = HfFileSystem(expand_info=True)
        self.path = DatasetPathHandler()

    def ls(self, target):
        target = self.path.to_string(target)
        for (attempt, delay) in enumerate(self.backoff, 1):
            try:
                yield from self.fs.ls(target)
                break
            except Exception as err:
                delay = retry_after(err) or delay
                Logger.error(
                    '%s: %s (attempt=%d, backoff=%ds)',
                    type(err).__name__,
                    ' '.join(str(err).split()),
                    attempt,
                    delay,
                )
            time.sleep(delay)

    def walk(self, target):
        for i in self.ls(target):
            name = i['name']
            if self.fs.isdir(name):
                for j in self.ls(name):
                    path = Path(j['name'])
                    if path.stem.startswith('samples_'):
                        date = j['last_commit'].date
                        yield Result(path, date)

#
#
#
def func(incoming, outgoing, args):
    fs = DatasetFileSystem(Backoff(args.backoff, 0.1))
    results = {}

    while True:
        dataset = incoming.get()
        Logger.info(dataset)

        results.clear()
        for i in fs.walk(dataset):
            key = repr(i)
            if key not in results or i > results[key]:
                results[key] = i

        outgoing.put(list(map(asdict, results.values())))

def records(args):
    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args,
    )

    with Pool(args.workers, func, initargs):
        models = ModelIterator(args.author)
        datasets = DatasetIterator(args.author)

        handle = ModelHandler(args.author)
        if args.exclude_flagged:
            handle = FlaggedHandler(handle, datasets)
        with ModelDatabase(args.database) as db:
            handle = DatabaseHandler(handle, datasets, db)

            jobs = 0
            for m in handle(models):
                outgoing.put(Path(m.id))
                jobs += 1

            for _ in range(jobs):
                results = incoming.get()
                yield from results

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--backoff', type=float, default=15)
    arguments.add_argument('--exclude-flagged', action='store_true')
    arguments.add_argument('--database', type=Path, required=True)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    fieldnames = [ x.name for x in fields(Result) ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records(args))
