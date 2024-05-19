import csv
import gzip
import itertools as it
from pathlib import Path
from argparse import ArgumentParser
from urllib.parse import urlparse
from multiprocessing import Pool, JoinableQueue

import pandas as pd

from mylib import Logger, FileChecksum, SimpleStorageWriter

def load(path):
    converters = {
        'date': pd.to_datetime,
        'value': float,
    }
    with gzip.open(path, mode='rt') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            row.update((x, y(row[x])) for (x, y) in converters.items())
            yield row

def func(queue, args):
    active = True
    with SimpleStorageWriter(args.bucket, args.chunk_size) as writer:
        while active:
            path = queue.get()
            if path is None:
                active = False
            else:
                Logger.info(path)
                for i in path.rglob('*.csv.gz'):
                    checksum = FileChecksum(i)
                    if not checksum:
                        Logger.error(i)
                        continue
                    with gzip.open(i, mode='rt') as fp:
                        reader = csv.DictReader(fp)
                        flushed = writer.write(reader)
                        if flushed:
                            Logger.warning('Flushed %s', flushed)
            queue.task_done()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--bucket', type=urlparse)
    arguments.add_argument('--index-path', type=Path)
    arguments.add_argument('--chunk-size', type=int, default=int(1e6))
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args,
    )

    with Pool(args.workers, func, initargs) as pool:
        phases = (
            args.index_path.iterdir(),
            it.repeat(None, pool._processes),
        )
        for jobs in phases:
            for j in jobs:
                queue.put(j)
            queue.join()
