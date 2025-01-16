import sys
import csv
import time
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from dataclasses import dataclass, asdict, fields
from multiprocessing import Pool, Queue

from huggingface_hub import HfApi, HfFileSystem

from mylib import Logger, Backoff, DatasetPathHandler

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
                    type(err),
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

def func(incoming, outgoing, args):
    fs = DatasetFileSystem(Backoff(args.backoff, 0.1))
    results = {}

    while True:
        dataset = incoming.get()
        Logger.info(dataset)

        results.clear()
        for i in fs.walk(dataset):
            key = repr(i)
            if key not in results or i < results[key]:
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
        api = HfApi()

        jobs = 0
        for i in api.list_datasets(author=args.author, search='-details'):
            outgoing.put(Path(i.id))
            jobs += 1

        for _ in range(jobs):
            results = incoming.get()
            yield from results

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--backoff', type=float, default=15)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    fieldnames = [ x.name for x in fields(Result) ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records(args))
