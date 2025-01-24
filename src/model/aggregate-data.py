from pathlib import Path
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, fields, astuple
from multiprocessing import Pool

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
                   .filter(items=rfields())
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

        writer = csv.DictWriter(sys.stdout, fieldnames=rfields())
        writer.writeheader()
        for _ in range(jobs):
            records = incoming.get()
            writer.writerows(records)
