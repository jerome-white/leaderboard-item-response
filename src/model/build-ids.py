import sys
import csv
from dataclasses import dataclass, fields

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
        return self.AuthorModel(*map(data.get, self.keys))

class DocumentHandler(VariableHandler):
    def extract(self, data):
        return data['document']

#
#
#
def scanf(fp):
    handlers = {
        'document_id': DocumentHandler(),
        'author_model_id': AuthorModelHandler(),
    }

    reader = csv.DictReader(fp)
    assert not any(x in reader.fieldnames for x in handlers)
    for row in reader:
        for (k, handle) in handlers.items():
            row[k] = handle(row)
        yield row

if __name__ == '__main__':
    writer = None
    for row in scanf(sys.stdin):
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row)
            writer.writeheader()
        writer.writerow(row)
