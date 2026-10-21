import sys
import csv
import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
from collections.abc import Iterator

import arviz as az
import cmdstanpy
import matplotlib
from arviz_base.labels import BaseLabeller

from mylib import Logger

class FlatLabeller(BaseLabeller):
    def make_label_vert(self, var_name, sel, isel):
        return self.make_label_flat(var_name, sel, isel)

@dataclass(frozen=True)
class Parameter:
    name: str
    index: int

    def __repr__(self):
        return f'{self.name}[{self.index}]'

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, parameter: str):
        (name, _, index) = parameter.partition('[')
        assert index.endswith(']')
        index = int(index[:-1])

        return cls(name, index)

def scanf(fp, posterior) -> Iterator[Parameter]:
    n_sample_dims = 2

    reader = csv.DictReader(sys.stdin)
    for row in reader:
        parameter = row['parameter']
        Logger.info('%s %f', parameter, float(row['ess_bulk']))

        param = Parameter.from_string(parameter)

        dvar = posterior.data_vars.get(param.name)
        if dvar is None:
            raise ValueError(f'Unknown parameter: {param}')

        if dvar.ndim > n_sample_dims:
            dim = dvar.dims[n_sample_dims]
            if param.index not in dvar.coords[dim].values:
                raise ValueError(f'Invalid index: {repr(param)}')

        yield param

if __name__ == '__main__':
    _ptypes = {
        'rank': az.plot_rank,
        'trace': az.plot_trace,
    }

    arguments = ArgumentParser()
    arguments.add_argument('--chains', type=Path)
    arguments.add_argument('--kind', choices=_ptypes, default='trace')
    arguments.add_argument('--output', type=Path)
    args = arguments.parse_args()

    posterior = cmdstanpy.from_csv(path=str(args.chains))
    idata = az.from_cmdstanpy(posterior=posterior)

    coords = cl.defaultdict(list)
    var_names = set()
    for param in scanf(sys.stdin, idata.posterior):
        key = f'{param}_dim_0'
        coords[key].append(param.index)
        var_names.add(param.name)

    plot = _ptypes.get(args.kind)(
        idata,
        var_names=list(var_names),
        coords=coords,
        labeller=FlatLabeller(),
        col_wrap=1,
    )
    plot.savefig(args.output)
