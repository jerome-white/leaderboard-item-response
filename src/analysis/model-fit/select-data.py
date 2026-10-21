import sys
import functools as ft
from argparse import ArgumentParser

import pandas as pd

@ft.singledispatch
def select(parameters: tuple):
    def do(x: pd.DataFrame):
        mask = df['parameter'].str.startswith(parameters, na=False)
        return df[mask]

    return do

@select.register
def _(parameters: list):
    return select(tuple(parameters))

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--worst-k', type=int)
    arguments.add_argument('--parameter', action='append')
    args = arguments.parse_args()

    df = pd.read_csv(sys.stdin)
    if args.parameter:
        df = df.pipe(select(args.parameter))
    if args.worst_k is not None:
        df = (df
              .sort_values(by=['ess_bulk', 'r_hat'], ascending=[True, False])
              .head(args.worst_k))
    df.to_csv(sys.stdout, index=False)
