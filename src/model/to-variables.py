import csv
import json
import functools as ft
import collections as cl
from pathlib import Path
from argparse import ArgumentParser

#
#
#
class MyEncoder(json.JSONEncoder):
    @ft.singledispatchmethod
    def default(self, o):
        return super().default(o)

    @default.register
    def _(self, o: set):
        return list(o)

#
#
#
class SourceHandler:
    def __call__(self, row):
        raise NotImplementedError()

class DocumentHandler(SourceHandler):
    def	__call__(self, row):
        return row['document']

class AuthorModelHandler(SourceHandler):
    _keys = (
        'author',
        'model',
    )

    def __call__(self, row):
        authmod = Path(*(map(row.get, self._keys)))
        return str(authmod)

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data-file', type=Path)
    args = arguments.parse_args()

    _handlers = {
        'document': DocumentHandler(),
        'author_model': AuthorModelHandler(),
    }

    corpus = cl.defaultdict(lambda: cl.defaultdict(set))
    with args.data_file.open() as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            for (name, handler) in _handlers.items():
                index = int(row[f'{name}_id'])
                corpus[name][index] = handler(row)

    print(json.dumps(corpus, cls=MyEncoder))
