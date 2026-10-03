import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import asdict
from multiprocessing import Pool, Queue

from mylib import Logger, Experiment

def func(incoming: Queue, outgoing: Queue, output: Path):
    while True:
        subject = incoming.get()

        experiment = Experiment('mmlu', subject, [subject])
        Logger.info(experiment)

        category = experiment.name.replace(' ', '-')
        out = (output
               .joinpath(experiment.benchmark, category, 'experiment')
               .with_suffix('.json'))
        dump = json.dumps(asdict(experiment), indent=2)

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w') as fp:
            print(dump, file=fp)

        outgoing.put(out)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args.output,
    )

    with Pool(args.workers, func, initargs):
        subjects = [
            'biology',
            'business',
            'chemistry',
            'computer science',
            'economics',
            'engineering',
            'health',
            'history',
            'law',
            'math',
            # 'other',
            'philosophy',
            'psychology',
            'physics',
        ]

        for s in subjects:
            outgoing.put(s)

        for _ in subjects:
            dst = incoming.get()
            print(dst)
