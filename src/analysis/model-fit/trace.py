import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
from collections.abc import Iterator

import arviz as az
import cmdstanpy

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

def parameters(params: list, posterior) -> Iterator[Parameter]:
    n_sample_dims = 2

    for p in params:
        param = Parameter.from_string(p)

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
    arguments.add_argument(
        '--parameter',
        action='append',
        help='"param[index]", where index is zero-based',
    )
    args = arguments.parse_args()

    posterior = cmdstanpy.from_csv(path=str(args.chains))
    idata = az.from_cmdstanpy(posterior=posterior)

    coords = cl.defaultdict(list)
    var_names = set()
    for p in parameters(args.parameter, idata.posterior):
        key = f'{p}_dim_0'
        coords[key].append(p.index)
        var_names.add(p.name)

    plot = _ptypes.get(args.kind)(
        idata,
        var_names=list(var_names),
        coords=coords,
    )
    plot.savefig(args.output)
