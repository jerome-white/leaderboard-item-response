import sys
import csv
from datetime import datetime
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Queue

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
def func(incoming, outgoing, args):
    while True:
        ev_set = incoming.get()
        Logger.info(ev_set)

        results = []

        ev_info = EvaluationInfo.from_evaluation_set(ev_set)
        with TemporaryDirectory(suffix=f'.{ev_set.uri.name}') as cache_dir:
            kwargs = {
                'cache_dir': cache_dir,
            }
            download_config = DownloadConfig(
                disable_tqdm=True,
                max_retries=args.max_retries,
                **kwargs,
            )
            try:
                ds = load_dataset(
                    str(ev_info),
                    ev_set.evaluation,
                    download_config=download_config,
                    **kwargs,
                )
                key = min(d_times(ds.keys()))
                values = extract(
                    ev_info,
                    key.to_datetime(),
                    ds.get(str(key)),
                )
                results.extend(map(asdict, values))
            except Exception as err:
                Logger.error(f'{ev_set}: Cannot retrieve data ({err})')

        outgoing.put(results)

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
