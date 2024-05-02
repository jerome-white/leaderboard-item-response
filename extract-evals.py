import sys
import csv
from datetime import datetime
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Queue

import pandas as pd
from datasets import DownloadConfig, load_dataset

from mylib import Logger, EvaluationSet, EvaluationInfo, LeaderboardResult

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
def extract(info, date, data):
    _metrics = (
        'em',
        'f1',
        'mc1',
        'mc2',
        'acc',
        'acc_norm',
    )
    kwargs = asdict(info)

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

#
#
#
def each(df):
    for i in df.itertuples(index=False):
        kwargs = i._asdict()
        yield EvaluationSet(**kwargs)

def func(incoming, outgoing, args):
    download_config = DownloadConfig(
        disable_tqdm=True,
        max_retries=args.max_retries,
    )

    while True:
        (group, df) = incoming.get()
        Logger.critical(group)

        for i in each(df):
            Logger.info(i)

            info = EvaluationInfo.from_evaluation_set(i)
            try:
                ds = load_dataset(
                    i.uri,
                    i.evaluation,
                    download_config=download_config,
                    streaming=True,
                )
                key = min(d_times(ds.keys()))
                values = extract(info, key.to_datetime(), ds.get(str(key)))
                outgoing.put(list(map(asdict, values)))
            except Exception as err:
                Logger.error(f'{i}: Cannot retrieve data ({err})')

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
        df = pd.read_csv(sys.stdin)
        jobs = 0
        for i in df.groupby('uri', order=False):
            outgoing.put(i)
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
