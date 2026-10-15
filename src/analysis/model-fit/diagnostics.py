import sys
from pathlib import Path
from argparse import ArgumentParser

import arviz as az
import cmdstanpy

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--chains', type=Path)
    arguments.add_argument('--rhat-threshold', type=float, default=1.1)
    args = arguments.parse_args()

    posterior = cmdstanpy.from_csv(path=str(args.chains))
    idata = az.from_cmdstanpy(posterior=posterior)

    df = (az
          .summary(idata, var_names=['alpha', 'beta', 'theta'])
          .assign(family=lambda x: x.index.str.split('[').str[0])
          .groupby('family')
          .agg(n=('r_hat', 'size'),
               bad=('r_hat', lambda x: x.gt(args.rhat_threshold).sum()),
               ess_bulk_min=('ess_bulk', 'min'),
               ess_bulk_median=('ess_bulk', 'median'))
          .assign(bad_pct=lambda x: x['bad'].mul(100).div(x['n'])))

    df.to_csv(sys.stdout)
