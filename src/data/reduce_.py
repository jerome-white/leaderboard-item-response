import sys
import csv
import gzip
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import fields, astuple
from multiprocessing import Pool

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
def func(path):
    Logger.debug(path)

    parser = SubmissionParser()
    with gzip.open(path, mode='rt') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            return parser(row)

    Logger.critical('removing empty file %s', path)
    path.unlink()

def scan(args):
    with Pool(args.workers) as pool:
        iterable = args.corpus.rglob('*.csv.gz')
        yield from filter(None, pool.imap_unordered(func, iterable))

def extract(db, fp):
    parser = SubmissionParser()
    reader = csv.DictReader(fp)
    for row in reader:
        (info, date) = parser(row)
        if info in db and db[info] <= date:
            Logger.warning('skipping %s', info.to_path())
            continue
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
    for row in extract(dict(scan(args)), sys.stdin):
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row)
            writer.writeheader()
        writer.writerow(row)
