import os
import sys
import csv
from hashlib import blake2b
from datetime import datetime
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Queue

from datasets import DownloadConfig, load_dataset

from mylib import (
    Logger,
    EvaluationSet,
    EvaluationInfo,
    LeaderboardValue,
    LeaderboardResult,
)

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
class MetricExtractor:
    _metrics = (
        'em',
        'f1',
        'mc1',
        'mc2',
        'acc',
        'acc_norm',
    )
    _digest_size = 16

    def __init__(self, ev_set):
        ev_info = EvaluationInfo.from_evaluation_set(ev_set)
        self.kwargs = asdict(ev_info)

    def __call__(self, dataset):
        key = min(d_times(dataset.keys()))
        date = key.to_datetime()

        for row in dataset.get(str(key)):
            prompt = self.encode(row['full_prompt'])
            for m in self.metrics(row):
                kwargs = asdict(m)
                yield LeaderboardResult(
                    date=date,
                    prompt=prompt,
                    **kwargs,
                    **self.kwargs,
                )

    def encode(self, value):
        message = blake2b(value.encode(), digest_size=self._digest_size)
        return message.hexdigest()

    def metrics(self, record):
        for metric in self._metrics:
            if metric in record:
                value = float(record[metric])
                yield LeaderboardValue(metric, value)

#
#
#
def func(incoming, outgoing, args):
    download_config = DownloadConfig(
        disable_tqdm=True,
        max_retries=args.max_retries,
        token=os.getenv('HF_BEARER_TOKEN'),
    )

    while True:
        ev_set = incoming.get()
        Logger.info(ev_set)

        extractor = MetricExtractor(ev_set)
        try:
            ds = load_dataset(
                ev_set.uri,
                ev_set.evaluation,
                download_config=download_config,
                streaming=True,
            )
            outgoing.put(list(map(asdict, extractor(ds))))
        except Exception as err:
            Logger.error('%s: %s (%s)', ev_set, err, type(err).__name__)
        finally:
            outgoing.put(None)

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--max-retries', type=int, default=5)
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
            ev_set = EvaluationSet(**row)
            outgoing.put(ev_set)
            jobs += 1

        writer = None
        while jobs:
            rows = incoming.get()
            if rows is None:
                jobs -= 1
            elif rows:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerows(rows)
