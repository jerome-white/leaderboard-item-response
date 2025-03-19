import sys
from pathlib import Path
from argparse import ArgumentParser

import pymc as pm
import pandas as pd
import pymc.sampling.jax as pmjax

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--seed', type=int)
    args = arguments.parse_args()

    df = pd.read_csv(sys.stdin, index_col=['author', 'model'])
    (persons, items) = df.shape

    if pmjax.jax.local_device_count() == 1:
        chain_method = 'vectorized'
    else:
        chain_method = 'parallel'

    with pm.Model() as model:
        # Priors
        alpha = pm.LogNormal('alpha', mu=0.5, sigma=1, shape=items)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=items)
        theta = pm.Normal('theta', mu=0, sigma=1, shape=persons)

        # Linear predictor
        eta = alpha[None, :] * (theta[:, None] - beta[None, :])

        # Likelihood
        y_obs = pm.Bernoulli(
            'y_obs',
            logit_p=eta,
            observed=df.to_numpy(),
        )

        trace = pmjax.sample_numpyro_nuts(
            chain_method=chain_method,
            random_seed=args.seed,
            # target_accept=0.9,
            idata_kwargs={
                'log_likelihood': False,
            }
        )
    trace.to_netcdf(args.output)
