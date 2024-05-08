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

from mylib import Logger, Backoff, DatasetPathHandler, hf_datetime

@dataclass
class Result:
    path: Path
    dt: datetime

    def __lt__(self, other):
        return self.dt < other.dt

class DatasetFileSystem:
    def __init__(self, token, backoff):
        self.fs = HfFileSystem(token=token)
        self.backoff = backoff

    def walk(self, target):
        for (i, j) in enumerate(self.backoff, 1):
            try:
                yield from self.fs.ls(target)
                break
            except HfHubHTTPError as err:
                Logger.error(
                    '%s: request=%s message=%s (attempt=%d, backoff=%ds)',
                    err,
                    err.request_id,
                    err.server_message,
                    i,
                    j,
                )
            time.sleep(j)

    def options(self, target):
        for i in self.walk(target):
            if i['type'] == 'directory':
                path = Path(i['name'])
                try:
                    dt = hf_datetime(path.name)
                except ValueError:
                    continue
                yield Result(path, dt)

    def results(self, target):
        for i in self.walk(target):
            yield i['name']

def func(incoming, outgoing, args):
    backoff = Backoff(args.backoff, 0.1)
    fs = DatasetFileSystem(os.getenv('HF_BEARER_TOKEN'), backoff)
    target = DatasetPathHandler()

    while True:
        dataset = incoming.get()
        Logger.info(dataset)

        results = []
        try:
            focus = min(fs.options(target.to_string(dataset)))
            results.extend(fs.results(target.to_string(focus.path)))
        except (ValueError, FileNotFoundError) as err:
            Logger.warning(f'{dataset}: {err}')
        outgoing.put(results)

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
