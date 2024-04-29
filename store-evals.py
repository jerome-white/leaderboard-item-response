import sys
import csv
import random
from argparse import ArgumentParser
from urllib.parse import urlparse
from multiprocessing import Pool, JoinableQueue

from mylib import Logger, CSVFileWriter, SimpleStorageWriter

def func(queue, args):
    Writer = SimpleStorageWriter if args.output.scheme else CSVFileWriter
    with Writer(args.output, args.chunk_size) as writer:
        while True:
            rows = queue.get()
            flushed = writer.write(rows)
            if flushed:
                Logger.info(f'Flushed {args.chunk_size}')
            queue.task_done()

def gather(fp):
    records = []
    assert args.chunk_size > 1
    limit = random.randint(1, args.chunk_size)

    reader = csv.DictReader(fp)
    for row in reader:
        records.append(row)
        if records > limit:
            yield records
            records = []
            limit = random.randint(1, args.chunk_size)

    if records:
        yield records

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=urlparse)
    arguments.add_argument('--chunk-size', type=int, default=int(1e5))
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args,
    )

    with Pool(args.workers, func, initargs):
        for rows in gather(sys.stdin):
            queue.put(rows)
        queue.join()
