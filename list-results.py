import sys
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import Pool, Queue

from huggingface_hub import HfFileSystem

from mylib import Logger, DatasetPathHandler, hf_datetime

@dataclass
class Result:
    path: Path
    dt: datetime

    def __lt__(self, other):
        return self.dt < other.dt

class DatasetFileSystem:
    def __init__(self):
        self.fs = HfFileSystem()

    def options(self, target):
        for i in self.fs.ls(target):
            if i['type'] == 'directory':
                path = Path(i['name'])
                try:
                    dt = hf_datetime(path.name)
                except ValueError:
                    continue
                yield Result(path, dt)

    def results(self, target):
        for i in self.fs.ls(target):
            yield i['name']

def func(incoming, outgoing):
    fs = DatasetFileSystem()
    target = DatasetPathHandler()

    while True:
        dataset = incoming.get()
        Logger.info(dataset)

        try:
            focus = min(fs.options(target.to_string(dataset)))
            results = list(fs.results(target.to_string(focus.path)))
        except (ValueError, FileNotFoundError) as err:
            Logger.error(f'{dataset}: {err}')
            results = None
        finally:
            outgoing.put(results)

if __name__ == '__main__':
    arguments = ArgumentParser()
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

        for _ in range(jobs):
            results = incoming.get()
            if results:
                print(*results, sep='\n')
