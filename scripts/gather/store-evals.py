import sys
import csv
from argparse import ArgumentParser
from urllib.parse import urlparse

from mylib import Logger, CSVFileWriter, SimpleStorageWriter

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=urlparse)
    arguments.add_argument('--chunk-size', type=int, default=int(1e5))
    args = arguments.parse_args()

    Writer = SimpleStorageWriter if args.output.scheme else CSVFileWriter
    with Writer(args.output, args.chunk_size) as writer:
        reader = csv.DictReader(sys.stdin)
        for row in reader:
            flushed = writer.write(row)
            if flushed:
                Logger.info(f'Flushed {args.chunk_size}')
