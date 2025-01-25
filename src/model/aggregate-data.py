import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd

from mylib import Logger

def func(args):
    (path, metric) = args
    Logger.info(path)

    mkey = 'metric'
    df = pd.read_csv(path, compression='gzip', memory_map=True)
    if metric not in df[mkey].values:
        Logger.error(
            '%s: metric \"%s\" not found (%s)',
            path,
            metric,
            ', '.join(df[mkey].unique())
        )
        return

    return (df
            .groupby(mkey, sort=False)
            .get_group(metric)
            .filter(items=[
                'author',
                'model',
                'document',
                'score',
            ])
            .to_dict(orient='records'))

def each(args):
    metric = {
        'arc': 'acc_norm',
        'bbh': 'acc_norm',
        'mmlu': 'acc',
        'gpqa': 'acc_norm',
        'musr': 'acc_norm',
        'math': 'exact_match',
        'gsm8k': 'exact_match',
        'ifeval': 'prompt_level_strict_acc',
    }[args.source.name]

    for p in args.source.rglob('*.csv.gz'):
        yield (p, metric)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        writer = None
        for i in pool.imap_unordered(func, each(args)):
            if i:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=i[0])
                    writer.writeheader()
                writer.writerows(i)
