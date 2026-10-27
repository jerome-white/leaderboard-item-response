import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from collections.abc import Iterator

import arviz as az
import pandas as pd
import cmdstanpy

def spread(da, window: int) -> pd.Series:
    """Per-draw dispersion across chains, smoothed over `window` draws
    and z-scored. High values mark draws where chains have wandered
    apart from one another; correlating this signal between two
    parameters shows whether they wander apart *at the same time*."""

    df = (pd
          .DataFrame(da.values.T, columns=da.coords['chain'].values)
          .rolling(window, min_periods=window)
          .mean()
          .dropna()
          .std(axis='columns'))

    return (df
            .sub(df.mean())
            .div(df.std()))

def scanf(fp, idata: az.InferenceData, window: int) -> Iterator[tuple]:
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

        yield (parameter, spread(dvar, window))

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--chains', type=Path)
    arguments.add_argument('--window', type=int, default=50)
    args = arguments.parse_args()

    posterior = cmdstanpy.from_csv(path=str(args.chains))
    idata = az.from_cmdstanpy(posterior=posterior)
    series = dict(scanf(sys.stdin, idata, args.window))

    df = (pd
          .DataFrame(series)
          .corr()
          .rename_axis(index='parameter'))
    df.to_csv(sys.stdout)
