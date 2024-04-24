import sys
import csv
from urllib.parse import urlparse, urlunparse
from argparse import ArgumentParser
from multiprocessing import Pool, JoinableQueue

import pandas as pd
import awswrangler as wr

from mylib import Logger

def func(queue, args):
    path = urlunparse(args.bucket)

    while True:
        records = queue.get()
        Logger.info(len(records))

        df = pd.DataFrame.from_records(records)
        wr.s3.to_parquet(
            df=df,
            path=path,
            dataset=True,
            use_threads=True,
            compression='gzip',
            partition_cols=[
                'evaluation',
                'author',
            ],
            dtype={
                'date': 'date',
            }
        )

        queue.task_done()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--bucket', type=urlparse)
    arguments.add_argument('--chunk-size', type=int, default=int(1e4))
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args,
    )

    with Pool(args.workers, func, initargs):
        reader = csv.DictReader(sys.stdin)
        records = []

        for row in reader:
            records.append(row)
            if len(records) >= args.chunk_size:
                queue.put(records)
                records = []
        if records:
            queue.put(records)

        queue.join()
