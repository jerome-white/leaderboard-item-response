import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from multiprocessing import Pool, JoinableQueue

from mylib import Logger, Experiment

def func(queue, args):
    while True:
        (name, subjects) = queue.get()
        e = Experiment('mmlu', name, subjects)
        Logger.info(e)

        category = e.name.replace(' ', '-')
        out = (args
               .output
               .joinpath(e.benchmark, category, 'experiment')
               .with_suffix('.json'))
        if out.exists():
            Logger.error(f'{out} exists')
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open('w') as fp:
                print(json.dumps(asdict(e), indent=2), file=fp)

        queue.task_done()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args,
    )

    with Pool(args.workers, func, initargs):
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
            queue.put((k, subjects))
            for s in subjects:
                queue.put((s, [s]))
        queue.join()
