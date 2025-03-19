import sys
import json
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data-file', type=Path)
    arguments.add_argument('--save-metadata', type=Path)
    args = arguments.parse_args()

    df = (pd
          .read_csv(args.data_file,
                    compression='gzip',
                    memory_map=True)
          .pivot(index=['author', 'model'],
                 columns='document',
                 values='score')
          .dropna(axis='columns'))

    iterable = zip(('persons', 'items'), (df.index, df.columns))
    meta = { x: y.to_list() for (x, y) in iterable }
    with args.save_metadata.open('w') as fp:
        print(json.dumps(meta, indent=2), file=fp)

    df.to_csv(sys.stdout)
