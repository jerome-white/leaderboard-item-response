import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd

from mylib import Logger

def func(incoming, outgoing, benchmark):
    items = [
        'author',
        'model',
        'document',
        'score',
    ]
    metric = {
        'arc': 'acc_norm',
        'bbh': 'acc_norm',
        'mmlu': 'acc',
        'gpqa': 'acc_norm',
        'musr': 'acc_norm',
        'math': 'exact_match',
        'gsm8k': 'exact_match',
        'ifeval': 'prompt_level_strict_acc',
    }[benchmark]

    while True:
        path = incoming.get()
        Logger.info(path)

        records = (pd
                   .read_csv(path, compression='gzip', memory_map=True)
                   .groupby('metric', sort=False)
                   .get_group(metric)
                   .filter(items=items)
                   .to_dict(orient='records'))

        outgoing.put(records)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args.source.name,
    )

    with Pool(args.workers, func, initargs):
        jobs = 0
        for i in args.source.rglob('*.csv.gz'):
            outgoing.put(i)
            jobs += 1

        writer = None
        for _ in range(jobs):
            records = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=records[0])
                writer.writeheader()
            writer.writerows(records)
