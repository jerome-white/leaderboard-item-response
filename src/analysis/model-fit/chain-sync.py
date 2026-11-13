import sys
import csv
from typing import TextIO
from pathlib import Path
from argparse import ArgumentParser
from collections.abc import Iterator

import arviz as az
import pandas as pd
import cmdstanpy

# Per-draw dispersion across chains, smoothed over `window`
# draws. High values mark draws where chains have wandered apart from
# one another; correlating the *rank* of this signal between two
# parameters shows whether they wander apart *at the same time*,
# without assuming a linear relationship or being thrown off by a
# handful of extreme draws.
class ChainDispersion:
    def __init__(self, window: int):
        self.window = window

    def __call__(self, da) -> pd.Series:
        return (pd
                .DataFrame(da.values.T, columns=da.coords['chain'].values)
                .rolling(self.window, min_periods=self.window)
                .mean()
                .dropna()
                .std(axis='columns'))

def scanf(
        fp: TextIO,
        idata, # : xarray.DataTree,
        spread: ChainDispersion,
) -> Iterator[tuple]:
    reader = csv.DictReader(fp)
    for row in reader:
        parameter = row['parameter']
        (name, _, index) = parameter.partition('[')

        dvar = idata.posterior.data_vars.get(name)
        if dvar is None:
            raise ValueError(f'Unknown parameter: {parameter}')

        if index:
            dim = dvar.dims[2]
            dvar = dvar.sel({dim: int(index[:-1])})

        yield (parameter, spread(dvar))

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--chains', type=Path)
    arguments.add_argument('--window', type=int, default=50)
    args = arguments.parse_args()

    spread = ChainDispersion(args.window)

    posterior = cmdstanpy.from_csv(path=str(args.chains))
    idata = az.from_cmdstanpy(posterior=posterior)

    series = scanf(sys.stdin, idata, spread)
    df = (pd
          .DataFrame(dict(series))
          .corr(method='spearman')
          .rename_axis(index='parameter'))
    df.to_csv(sys.stdout)
