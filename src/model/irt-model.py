from pathlib import Path
from argparse import ArgumentParser

import pymc as pm
import pandas as pd

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data-file', type=Path)
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--seed', type=int)
    args = arguments.parse_args()

    df = (pd
          .read_csv(args.data_file,
                    compression='gzip',
                    memory_map=True)
          .pivot(index=['author', 'model'],
                 columns='document',
                 values='score')
          .dropna(axis='columns'))
    (persons, items) = df.shape

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

        trace = pm.sampling_jax.sample_numpyro_nuts(
            # chain_method='vectorized',
            random_seed=args.seed,
            idata_kwargs={
                'log_likelihood': False,
            }
        )
    trace.to_netcdf(args.output)
