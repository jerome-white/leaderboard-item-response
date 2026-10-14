#
# First pass: load a subject's posterior draws through ArviZ and
# recompute Rhat/ESS, as a cross-check against what stansummary (and
# the bin/*-check.sh awk scripts) already report for the same chains.
# Deliberately basic - this is step 1 of #66, not a posterior
# predictive check yet.
#

import re
from argparse import ArgumentParser
from pathlib import Path

import arviz as az
import pandas as pd
import cmdstanpy

def family(name: str) -> str:
    return re.match(r'^[a-z]+', name).group()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path, required=True,
                            help='Directory containing cmdstan chain_*.csv files')
    arguments.add_argument('--rhat-threshold', type=float, default=1.1)
    args = arguments.parse_args()

    chains = sorted(args.output.glob('chain_*.csv'))
    fit = cmdstanpy.from_csv(path=[str(c) for c in chains])
    idata = az.from_cmdstanpy(posterior=fit)

    summary = az.summary(idata, var_names=['alpha', 'beta', 'theta'])
    summary['family'] = summary.index.map(family)

    report = summary.groupby('family').agg(
        n=('r_hat', 'size'),
        bad=('r_hat', lambda x: (x > args.rhat_threshold).sum()),
        ess_bulk_min=('ess_bulk', 'min'),
        ess_bulk_median=('ess_bulk', 'median'),
    )
    report['bad_pct'] = 100 * report['bad'] / report['n']

    pd.set_option('display.float_format', '{:.1f}'.format)
    print(report[['n', 'bad', 'bad_pct', 'ess_bulk_min', 'ess_bulk_median']])
