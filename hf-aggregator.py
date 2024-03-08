import sys
import csv
import itertools as it
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Queue

from datasets import (
    DownloadConfig,
    load_dataset,
    disable_progress_bars,
    get_dataset_config_names,
)
from huggingface_hub import DatasetFilter, list_datasets

from logutils import Logger

#
#
#
@dataclass
class DataSetKey:
    key: str
    value: datetime

    def __str__(self):
        return self.key

    def __lt__(self, other):
        return self.value < other.value

@dataclass
class DataSetInfo:
    path: Path
    name: str

    def __str__(self):
        return f'{self.path} {self.name}'

    def describe(self):
        for (i, word) in enumerate(self.path.name.split('__')):
            if not i:
                (_, word) = word.split('_')
            yield word

@dataclass
class DataSetDetails:
    author: str
    model: str
    task: str

    def __init__(self, info):
        (self.author, self.model) = info.describe()
        (_, self.task) = info.name.split('_', maxsplit=1)

#
#
#
class MetricExtractor:
    @staticmethod
    def items(keys):
        for i in keys:
            try:
                value = datetime.strptime(i, '%Y_%m_%dT%H_%M_%S.%f')
            except ValueError:
                continue
            yield DataSetKey(i, value)

    def __call__(self, data):
        key = min(self.items(data.keys()))
        yield from self.get(data[str(key)])

    def get(self, data):
        raise NotImplementedError()

class SimpleExtractor(MetricExtractor):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def get(self, data):
        yield from data[self.key]

class ARC(SimpleExtractor):
    def __init__(self):
        super().__init__('acc_norm')

class HellaSwag(SimpleExtractor):
    def __init__(self):
        super().__init__('acc_norm')

class TruthfulQA(SimpleExtractor):
    def __init__(self):
        super().__init__('mc2')

class MMLU(SimpleExtractor):
    def __init__(self):
        super().__init__('acc')

class GSM8K(SimpleExtractor):
    def __init__(self):
        super().__init__('acc')

class Winogrande(MetricExtractor):
    def get(self, data):
        metrics = data['metrics']
        yield from it.chain.from_iterable(x.values() for x in metrics)

#
#
#
def func(incoming, outgoing):
    disable_progress_bars()

    _extractors = {
        'arc': ARC(),
        'hellaswag': HellaSwag(),
        'hendrycksTest': MMLU(),
        'truthfulqa': TruthfulQA(),
        'winogrande': Winogrande(),
        'gsm8k': GSM8K(),
    }

    while True:
        info = incoming.get()
        Logger.info(info)

        details = DataSetDetails(info)
        body = asdict(details)

        results = []
        (task, *_) = details.task.split('_')
        if task in _extractors:
            extractor = _extractors[task]
            with TemporaryDirectory() as cache_dir:
                data = load_dataset(
                    str(info.path),
                    info.name,
                    cache_dir=cache_dir,
                )
                results.extend(dict(body, correct=x) for x in extractor(data))
        else:
            Logger.error(f'Unrecognized task: "{task}" ({details.task})')

        outgoing.put(results)

def ls(author):
    filter_ = DatasetFilter(author=author)
    for i in list_datasets(filter=filter_):
        dataset = Path(i.id)
        if dataset.name.startswith('details_'):
            yield dataset

def pull(datasets):
    with TemporaryDirectory() as cache_dir:
        download_config = DownloadConfig(
            cache_dir=cache_dir,
            delete_extracted=True,
        )

        for d in datasets:
            try:
                configs = get_dataset_config_names(
                    str(d),
                    download_config=download_config,
                )
            except ValueError as err:
                Logger.warning(f'{d} Cannot get config names')
                continue

            for c in configs:
                yield DataSetInfo(d, c)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
    )

    with Pool(args.workers, func, initargs):
        jobs = 0
        for i in pull(ls(args.author)):
            outgoing.put(i)
            jobs += 1

        writer = None
        for _ in range(jobs):
            rows = incoming.get()
            if rows:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerow(rows)
