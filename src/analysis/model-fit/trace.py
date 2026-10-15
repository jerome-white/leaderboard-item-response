#
# Trace/rank plots for an explicit, named parameter (or parameters) -
# no auto-selection of "interesting" ones here; that's a separate
# concern (diagnostics.py's default/raw output is sortable by
# ess_bulk to find a candidate, e.g. `sort -t, -k6 -n`).
#

import re
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

import arviz as az
import cmdstanpy
import matplotlib
matplotlib.use('Agg')

def parse(parameter: str) -> tuple[str, int]:
    (name, index) = re.match(r'^([a-z]+)\[(\d+)\]$', parameter).groups()
    return (name, int(index))

PLOTS = {
    'trace': az.plot_trace,
    'rank': az.plot_rank,
}

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--chains', type=Path, required=True)
    arguments.add_argument('--parameter', action='append', required=True,
                            help="e.g. --parameter 'beta[113]' (repeatable, "
                                 "0-indexed to match diagnostics.py's labels)")
    arguments.add_argument('--kind', default='trace', choices=PLOTS)
    arguments.add_argument('--output', type=Path, required=True)
    args = arguments.parse_args()

    posterior = cmdstanpy.from_csv(path=str(args.chains))
    idata = az.from_cmdstanpy(posterior=posterior)

    coords = defaultdict(list)
    for p in args.parameter:
        (name, index) = parse(p)
        coords[f'{name}_dim_0'].append(index)

    plot = PLOTS[args.kind](
        idata,
        var_names=list({parse(p)[0] for p in args.parameter}),
        coords=dict(coords),
    )
    plot.savefig(args.output)
