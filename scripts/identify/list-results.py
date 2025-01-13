import os
import sys
import time
import itertools as it
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import Pool, Queue

from huggingface_hub import HfFileSystem
from huggingface_hub.utils import HfHubHTTPError

from mylib import Logger, Backoff, DatasetPathHandler

@dataclass
class Result:
    path: Path
    dt: datetime

    def __repr__(self):
        (*prefix, _) = self.path.stem.split('_')
        return str(Path(*prefix))

    def __lt__(self, other):
        return self.dt < other.dt

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
                        yield Result(path, j['last_commit'].date)

def func(incoming, outgoing, args):
    fs = DatasetFileSystem(Backoff(args.backoff, 0.1))
    results = {}

    while True:
        dataset = incoming.get()
        Logger.info(dataset)

        results.clear()
        for i in fs.walk(dataset):
            key = repr(i)
            if key not in results or results[key] > i:
                results[key] = i

        outgoing.put([ x.path for x in results.values() ])

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--backoff', type=float, default=15)
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

        for _ in range(jobs):
            results = incoming.get()
            if results:
                print(*results, sep='\n')
