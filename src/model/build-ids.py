import sys
import csv

from mylib import Logger, DocumentHandler, AuthorModelHandler

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
