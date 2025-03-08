import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import asdict
from multiprocessing import Pool

from mylib import Logger, Experiment

def func(args):
    (name, subjects, output) = args

    e = Experiment('mmlu', name, subjects)
    Logger.info(e)

    category = e.name.replace(' ', '-')
    out = (output
           .joinpath(e.benchmark, category, 'experiment')
           .with_suffix('.json'))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as fp:
        print(json.dumps(asdict(e), indent=2), file=fp)

    return out

def each(args):
    categories = {
        'humanities': [
            'history',
            'law',
            'philosophy',
        ],
        'social science': [
            'economics',
            'psychology',
        ],
        'natural science': [
            'biology',
            'chemistry',
            'physics',
        ],
        'formal science': [
            'computer science',
            'engineering',
            'math',
        ],
        'applied science': [
            'business',
            'health',
        ],
        # 'other',
    }

    for (k, subjects) in categories.items():
        yield (k, subjects, args.output)
        for s in subjects:
            yield (s, [s], args.output)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        for i in pool.imap_unordered(func, each(args)):
            print(i)
