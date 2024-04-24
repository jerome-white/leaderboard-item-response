import sys
import csv
from datetime import datetime
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Queue

from datasets import load_dataset
from huggingface_hub.utils import HfHubHTTPError

from mylib import Logger, EvaluationSet, EvaluationInfo

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

    def to_datetime(self):
        return self.value

def d_times(values):
    for i in values:
        try:
            v = datetime.strptime(i, '%Y_%m_%dT%H_%M_%S.%f')
        except ValueError:
            continue

        yield DataSetKey(i, v)

#
#
#
@dataclass
class LeaderboardResult:
    date: datetime
    author: str
    model: str
    evaluation: str
    prompt: str
    metric: str
    value: float

#
#
#
def extract(info, date, data):
    _metrics = (
        'f1',
        'mc1',
        'mc2',
        'acc',
        'acc_norm',
    )

    kwargs = asdict(info)
    kwargs.pop('uri')

    for row in data:
        prompt = row['hashes']['full_prompt']
        for metric in _metrics:
            if metric in row:
                value = float(row[metric])
                yield LeaderboardResult(
                    date=date,
                    prompt=prompt,
                    metric=metric,
                    value=value,
                    **kwargs,
                )

def retrieve(info, retries):
    suffix = f'.{info.uri.name}'
    with TemporaryDirectory(suffix=suffix) as cache_dir:
        path = str(info)
        for i in range(retries):
            try:
                return load_dataset(path, info.evaluation, cache_dir=cache_dir)
            except HfHubHTTPError as err:
                Logger.warning(f'{i}: {err}')

    raise ImportError(info)

#
#
#
def func(incoming, outgoing, args):
    while True:
        ev_set = incoming.get()
        Logger.info(ev_set)

        results = []
        ev_info = EvaluationInfo.from_evaluation_set(ev_set)
        try:
            data = retrieve(ev_info, args.retries)
            key = min(d_times(data.keys()))
            values = extract(ev_info, key.to_datetime(), data.get(str(key)))
            results.extend(map(asdict, values))
        except ImportError as err:
            Logger.exception(f'Cannot retrieve results: {err}')

        outgoing.put(results)

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--retries', type=int, default=5)
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
        jobs = 0
        reader = csv.DictReader(sys.stdin)
        for row in reader:
            outgoing.put(EvaluationSet(**row))
            jobs += 1

        writer = None
        for _ in range(jobs):
            rows = incoming.get()
            if rows:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerows(rows)
