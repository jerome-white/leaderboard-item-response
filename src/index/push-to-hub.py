import json
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, asdict
from multiprocessing import Pool, JoinableQueue

from datasets import Dataset, disable_progress_bars

from mylib import Logger

@dataclass(frozen=True)
class Document:
    doc: str
    info: dict

def reader(source):
    for s in source.iterdir():
        assert s.suffix.endswith('json')
        data = json.loads(s.read_text())
        for i in data.items():
            document = Document(*i)
            yield asdict(document)

def func(queue, prefix):
    disable_progress_bars()

    while True:
        path = queue.get()
        target = f'{prefix}-{path.name}'

        Logger.info('%s -> %s', path, target)
        with TemporaryDirectory() as cache_dir:
            dataset = Dataset.from_generator(
                reader,
                cache_dir=cache_dir,
                gen_kwargs={
                    'source': path,
                },
            )
            dataset.push_to_hub(target)
        queue.task_done()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--documents', type=Path)
    arguments.add_argument('--dataset-prefix', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args.dataset_prefix,
    )

    with Pool(args.workers, func, initargs):
        for i in args.documents.iterdir():
            queue.put(i)
        queue.join()
