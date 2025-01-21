import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, fields
from multiprocessing import Pool

from mylib import Logger

@dataclass(frozen=True)
class AuthorModel:
    author: str
    model: str

def scanf(path):
    db = {}
    keys = tuple(x.name for x in fields(AuthorModel))
    author_model_id = 'author_model_id'

    for i in path.iterdir():
        with i.open() as fp:
            reader = csv.DictReader(fp)
            assert author_model_id not in reader.fieldnames
            for row in reader:
                authmod = AuthorModel(*map(row.get, keys))
                row[author_model_id] = db.setdefault(authmod, len(db) + 1)

                yield row

def func(path):
    Logger.info(path)

    writer = None
    with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as fp:
        for row in scanf(path):
            if writer is None:
                writer = csv.DictWriter(fp, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)

    return fp.name

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        iterable = (Path(x.rstrip()) for x in sys.stdin)
        for i in pool.imap_unordered(func, iterable):
            print(i)
