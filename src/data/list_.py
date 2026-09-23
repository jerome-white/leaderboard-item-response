import sys
import csv
import time
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from dataclasses import dataclass, asdict, fields, replace
from multiprocessing import Pool, Queue

from datasets import load_dataset
from huggingface_hub import HfApi, HfFileSystem

from mylib import Dataset, Logger, Backoff, DatasetPathHandler

class ModelIterator:
    _dtype = '-details'

    def __init__(self, author):
        self.author = author
        self.api = HfApi()

    def __iter__(self):
        datasets = self.api.list_datasets(
            author=self.author,
            search=self._dtype,
        )
        for info in datasets:
            ds = Dataset.from_leaderboard(info.id, self.author)
            if self.is_legal(ds):
                yield info

    def is_legal(self, dataset):
        raise NotImplementedError()

class AllModels(ModelIterator):
    def is_legal(self, dataset):
        return True

class UnflaggedModels(ModelIterator):
    @staticmethod
    def flagged(dataset):
        for row in load_dataset(str(dataset), split='train'):
            if row['Flagged']:
                yield Dataset.from_fullname(row['fullname'])

    def __init__(self, author):
        super().__init__(author)
        dataset = Dataset(self.author, 'contents')
        self.datasets = set(self.flagged(dataset))

    def is_legal(self, dataset):
        name = dataset.name.removesuffix(self._dtype)
        ds = replace(dataset, name=name)

        return ds not in self.datasets

#
#
#
@dataclass
class Result:
    path: Path
    date: datetime

    def __repr__(self):
        (*prefix, _) = self.path.stem.split('_')
        return str(Path(*prefix))

    def __lt__(self, other):
        return self.date < other.date

class DatasetFileSystem:
    def __init__(self, backoff):
        self.backoff = backoff
        self.fs = HfFileSystem()
        self.path = DatasetPathHandler()

    def ls(self, target):
        target = self.path.to_string(target)
        for (i, j) in enumerate(self.backoff, 1):
            try:
                yield from self.fs.ls(target)
                break
            except Exception as err:
                Logger.error(
                    '%s: %s (attempt=%d, backoff=%ds)',
                    type(err).__name__,
                    err,
                    i,
                    j,
                )
            time.sleep(j)

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
        Models = UnflaggedModels if args.exclude_flagged else AllModels
        models = Models(args.author)

        jobs = 0
        for m in models:
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
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    fieldnames = [ x.name for x in fields(Result) ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records(args))
