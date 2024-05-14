import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd

from mylib import Logger, FileChecksum

def func(incoming, outgoing):
    while True:
        path = incoming.get()
        Logger.info(path)

        for i in path.rglob('*.csv.gz'):
            checksum = FileChecksum(i)
            if not checksum:
                Logger.error(i)
                continue
            df = (pd
                  .read_csv(i, compression='gzip')
                  .to_dict(orient='records'))
            outgoing.put(df)

        outgoing.put(None)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--index-path', type=Path)
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
        for i in args.index_path.iterdir():
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
