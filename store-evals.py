import sys
import csv
import functools as ft
from urllib.parse import urlparse
from argparse import ArgumentParser

from mylib import CSVWriter, SimpleStorageWriter

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--bucket', type=urlparse)
    arguments.add_argument('--chunk-size', type=int, default=int(1e4))
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    reader = csv.DictReader(sys.stdin)
    if args.bucket:
        Writer = ft.partial(SimpleStorageWriter, bucket=args.bucket)
    else:
        Writer = ft.partial(CSVWriter, fp=sys.stdout)

    with Writer(chunk_size=args.chunk_size) as writer:
        for row in reader:
            writer.write(row)
