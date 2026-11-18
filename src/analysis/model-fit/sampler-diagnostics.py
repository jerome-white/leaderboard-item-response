import sys
import functools as ft
from pathlib import Path
from argparse import ArgumentParser

import arviz as az
import pandas as pd
import cmdstanpy

@ft.singledispatch
def scanf(chains: str) -> pd.DataFrame:
    posterior = cmdstanpy.from_csv(path=chains)
    idata = az.from_cmdstanpy(posterior=posterior)

    bfmi = az.bfmi(idata)['energy']
    return pd.DataFrame([{
        'divergences': int(idata.sample_stats['diverging'].sum()),
        'bfmi_min': float(bfmi.min()),
        'bfmi_max': float(bfmi.max()),
    }])

@scanf.register
def _(chains: Path) -> pd.DataFrame:
    return scanf(str(chains))

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--chains', type=Path)
    args = arguments.parse_args()

    df = scanf(args.chains)
    df.to_csv(sys.stdout, index=False)
