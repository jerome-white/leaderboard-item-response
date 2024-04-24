import sys
import csv
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import fields, asdict
from multiprocessing import Pool, Queue

from requests import ConnectionError
from datasets import DownloadConfig, get_dataset_config_names
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

from mylib import Logger, EvaluationSet

#
#
#
def retrieve(path):
    with TemporaryDirectory() as cache_dir:
        dc = DownloadConfig(cache_dir=cache_dir, disable_tqdm=True)
        for _ in range(args.retries):
            try:
                return get_dataset_config_names(path, download_config=dc)
            except ConnectionError:
                continue
            except (ValueError, EmptyDatasetError) as err:
                break

    raise ImportError()

def func(incoming, outgoing, args):
    while True:
        path = incoming.get()
        Logger.info(path)

        records = []
        try:
            for i in retrieve(path):
                if i.startswith('harness_'):
                    ev_set = EvaluationSet(path, i)
                    records.append(asdict(ev_set))
        except ImportError as err:
            Logger.error(f'{path}: Cannot get config names')
        finally:
            outgoing.put(records)

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--retries', type=int, default=3)
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
        api = HfApi()
        for i in api.list_datasets(author=args.author, search='details_'):
            if i.id.casefold().find('flagged') >= 0:
                Logger.warning(f'Flagged: {i.id}')
                continue
            outgoing.put(i.id)
            jobs += 1

        writer = None
        for _ in range(jobs):
            rows = incoming.get()
            if rows:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                    writer.writeheader()
                writer.writerows(rows)
