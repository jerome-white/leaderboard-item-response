import sys
import csv
from argparse import ArgumentParser

from mylib import Logger, DocumentHandler

def scanf(fp, exclude: set):
    reader = csv.DictReader(fp)
    handle = DocumentHandler()

    for row in reader:
        doc_id = handle(row)
        if doc_id not in exclude:
            yield row

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--exclude', type=int, action='append')
    args = arguments.parse_args()

    writer = None
    for row in scanf(sys.stdin, set(args.exclude)):
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row)
            writer.writeheader()
        writer.writerow(row)
