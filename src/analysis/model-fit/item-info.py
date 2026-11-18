import sys
import json
from typing import Iterable
from pathlib import Path

import pandas as pd

def scanf(fp) -> Iterable[tuple]:
    data = json.load(fp)
    trx = (
        ('item', 'q_i'),
        ('y', 'y'),
    )

    for (k, v) in trx:
        yield (k, data[v])

def info(x: pd.DataFrame) -> pd.Series:
    rate = x['rate']
    return rate.mul(rate.rsub(1)) # peaks at 0.5, ->0 at floor/ceiling

if __name__ == '__main__':
    df = (pd
          .DataFrame(dict(scanf(sys.stdin)))
          .groupby('item')['y']
          .agg(rate='mean', n='size')
          .assign(info=info)
          .rename_axis(index='item'))
    df.to_csv(sys.stdout)
