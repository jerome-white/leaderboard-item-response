import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass

import arviz as az
import cmdstanpy

@dataclass(frozen=True)
class Parameter:
    name: str
    index: int

    @classmethod
    def from_string(cls, parameter: str):
        (name, _, index) = parameter.partition('[')
        assert index.endswith(']')
        index = int(index[:-1])

        return cls(name, index)

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

    coords = cl.defaultdict(list)
    var_names = set()
    for p in args.parameter:
        param = Parameter.from_string(p)

        key = f'{param.name}_dim_0'
        coords[key].append(param.index)
        var_names.add(param.name)

    posterior = cmdstanpy.from_csv(path=str(args.chains))
    idata = az.from_cmdstanpy(posterior=posterior)

    plot = _ptypes.get(args.kind)(
        idata,
        var_names=list(var_names),
        coords=coords,
    )
    plot.savefig(args.output)
