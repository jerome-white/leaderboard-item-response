import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import fields, astuple
from multiprocessing import Pool, Queue

import pandas as pd

from mylib import Logger, SubmissionInfo

#
#
#
class SubmissionParser:
    _keys = [ x.name for x in fields(SubmissionInfo) ]

    def __call__(self, data):
        info = SubmissionInfo(*map(data.get, self._keys))
        date = pd.to_datetime(data['date'])
        return (info, date)

#
#
#
def func(incoming, outgoing):
    parser = SubmissionParser()

    while True:
        path = incoming.get()
        Logger.info(path)

        with path.open() as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                outgoing.put(parser(row))
                break

def scan(args):
    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
    )

    with Pool(args.workers, func, initargs):
        jobs = 0
        for i in args.corpus.rglob('*.csv'):
            outgoing.put(i)
            jobs += 1

        for _ in range(jobs):
            result = incoming.get()
            yield result

def extract(db, fp):
    parser = SubmissionParser()

    reader = csv.DictReader(fp)
    for row in reader:
        (info, date) = parser(row)
        info = astuple(info)
        if info not in db or db[info] > date:
            yield row

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--corpus', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    writer = None
    for i in extract(dict(scan(args)), sys.stdin):
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=i)
            writer.writeheader()
        writer.writerow(i)
