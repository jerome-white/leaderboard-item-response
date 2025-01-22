import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, fields
from multiprocessing import Pool

from mylib import Logger

#
#
#
class VariableHandler:
    def __init__(self):
        self.db = {}

    def __call__(self, data):
        return self.db.setdefault(self.extract(data), len(self.db) + 1)

    def extract(self, data):
        raise NotImplementedError()

class AuthorModelHandler(VariableHandler):
    @dataclass(frozen=True)
    class AuthorModel:
        author: str
        model: str

    def __init__(self):
        super().__init__()
        self.keys = tuple(x.name for x in fields(self.AuthorModel))

    def extract(self, data):
        return self.AuthorModel(*map(data.get, keys))

class DocumentHandler(VariableHandler):
    def extract(self, data):
        return data['document']

#
#
#
class DocuScan:
    def __init__(self):
        self.handlers = {
            'document_id': DocumentHandler(),
            'author_model_id': AuthorModelHandler(),
        }

    def __call__(self, path):
        for row in self.scanf(path):
            for (k, handle) in self.handlers.items():
                row[k] = handle(row)
            yield row

    def scanf(self, path):
        for i in path.iterdir():
            with i.open() as fp:
                reader = csv.DictReader(fp)
                assert not any(x in reader.fieldnames for x in self.handlers)
                yield from reader

def func(incoming, outgoing):
    scanner = DocuScan()

    while True:
        path = incoming.get()
        Logger.info(path)

        writer = None
        with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as fp:
            for row in scanner(path):
                if writer is None:
                    writer = csv.DictWriter(fp, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
            outgoing.put(fp.name)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        iterable = (Path(x.rstrip()) for x in sys.stdin)
        for i in pool.imap_unordered(func, iterable):
            print(i)
