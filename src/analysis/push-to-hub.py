import sys
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory

import pandas as pd
from datasets import Dataset, disable_progress_bars

from mylib import Logger

def reader(fp, chunksize):
    with pd.read_csv(fp, chunksize=chunksize) as reader:
        for df in reader:
            Logger.info('[%d, %d]', df.index.min(), df.index.max())
            for i in df.itertuples(index=False):
                yield i._asdict()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--split')
    arguments.add_argument('--target', type=Path)
    arguments.add_argument('--window', type=int, default=int(1e5))
    args = arguments.parse_args()

    disable_progress_bars()
    with TemporaryDirectory() as cache_dir:
        dataset = Dataset.from_generator(
            reader,
            cache_dir=cache_dir,
            gen_kwargs={
                'fp': sys.stdin,
                'chunksize': args.window,
            },
            split=args.split,
        )
        dataset.push_to_hub(str(args.target))
