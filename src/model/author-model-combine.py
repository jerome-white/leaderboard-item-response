import csv
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, fields
from multiprocessing import Pool, JoinableQueue

from mylib import Logger

@dataclass(frozen=True)
class AuthorModel:
    author: str
    model: str

def scanf(path):
    db = {}
    keys = tuple(x.name for x in fields(AuthorModel))

    for i in path.iterdir():
        with i.open() as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                am = AuthorModel(*map(row.get, keys))
                author_model_id = db.setdefault(am, len(db))
                yield dict(row, author_model_id=author_model_id)

def func(queue, target):
    while True:
        path = queue.get()
        Logger.info(path)

        (*_, name, category) = path.parts
        dst = (target
               .joinpath(name, category, 'data')
               .with_suffix('.csv'))
        dst.parent.mkdir(parents=True)

        writer = None
        with dst.open('w') as fp:
            for row in scanf(path):
                if writer is None:
                    writer = csv.DictWriter(fp, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)

        queue.task_done()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--target', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args.target,
    )

    with Pool(args.workers, func, initargs):
        for i in args.source.iterdir():
            for j in i.iterdir():
                queue.put(j)
        queue.join()
