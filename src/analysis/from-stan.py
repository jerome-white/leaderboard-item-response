import sys
import csv
import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, asdict, fields
from multiprocessing import Pool, Queue

from mylib import Logger

#
#
#
@dataclass
class StanParameter:
    parameter: str
    index: int

    def __init__(self, value):
        (self.parameter, index) = value.split('.')
        self.index = int(index)

@dataclass
class _Sample:
    chain: int
    index: int

@dataclass
class Sample(_Sample):
    data: dict

    def __iter__(self):
        yield from self.data.items()

@dataclass
class Record(_Sample):
    parameter: str
    source: str
    value: float

#
#
#
class ParameterIndex:
    _variables = {
        'alpha': 'document',
        'beta': 'document',
        'theta': 'author_model',
    }

    def __init__(self, path):
        self.db = json.loads(path.read_text())

    def __getitem__(self, item):
        key = self._variables[item.parameter]
        return self.db[key][item.index]

    def is_parameter(self, value):
        return any(map(value.startswith, self._variables))

#
#
#
class DataIterator:
    def __init__(self, root, chunksize):
        self.root = root
        self.chunksize = chunksize
        self.windows = 0

    def __len__(self):
        return self.windows

    def __iter__(self):
        window = []

        for s in self.scanf():
            window.append(s)
            if len(window) >= self.chunksize:
                yield window
                self.windows += 1
                window = []

        if window:
            yield window
            self.windows += 1

    def scanf(self):
        for p in self.root.iterdir():
            (*_, chain) = p.stem.split('_')
            chain = int(chain)
            with p.open() as fp:
                reader = csv.DictReader(fp)
                for idx_data in enumerate(reader):
                    yield Sample(chain, *idx_data)

#
#
#
def func(incoming, outgoing, args):
    p_index = ParameterIndex(args.parameters)

    while True:
        samples = incoming.get()
        Logger.info(len(samples))

        records = []
        for s in samples:
            for (k, value) in s:
                if p_index.is_parameter(k):
                    s_param = StanParameter(k)
                    records.append(Record(
                        s.chain,
                        s.index,
                        s_param.parameter,
                        p_index[s_param],
                        value,
                    ))

        outgoing.put(records)

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--stan-output', type=Path)
    arguments.add_argument('--parameters', type=Path)
    arguments.add_argument('--chunksize', type=int, default=int(1e5))
    # arguments.add_argument('--with-sampler-info', action='store_true')
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args,
    )

    with Pool(args.workers, func, initargs):
        reader = DataIterator(args.stan_output, args.chunksize)
        for i in reader:
            outgoing.put(i)

        fieldnames = [ x.name for x in fields(Record) ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(len(reader)):
            rows = incoming.get()
            writer.writerows(map(asdict, rows))
