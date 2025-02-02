import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory

import pandas as pd
from datasets import Dataset

def reader(fp, window):
    with pd.read_csv(sys.stdin, iterator=True) as reader:
        df = reader.get_chunk(window)
        for i in df.itertuples(index=False):
            yield i._asdict()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--target', type=Path)
    arguments.add_argument('--window', type=int, default=int(1e5))
    args = arguments.parse_args()

    with TemporaryDirectory() as cache_dir:
        dataset = Dataset.from_generator(
            reader,
            cache_dir=cache_dir,
            gen_kwargs={
                'fp': sys.stdin,
                'window': args.window,
            },
        )
        dataset.push_to_hub(str(args.target))
