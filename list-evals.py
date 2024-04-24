import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, fields, asdict
from multiprocessing import Pool

from datasets import DownloadConfig, get_dataset_config_names
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi

from mylib import Logger, EvaluationSet

#
#
#
def func(dataset):
    Logger.info(dataset)

    with TemporaryDirectory() as cache_dir:
        dc = DownloadConfig(cache_dir=cache_dir)
        try:
            names = get_dataset_config_names(dataset, download_config=dc)
        except (ValueError, ConnectionError, EmptyDatasetError) as err:
            names = None
            Logger.exception('[{}] Cannot get config names: {} ({})'.format(
                dataset,
                type(err).__name__,
                err,
            ))

    records = []
    if names is not None:
        for i in names:
            if i.startswith('harness_'):
                ev_set = EvaluationSet(dataset, i)
                records.append(ev_set)

    return records

def ls(author):
    api = HfApi()
    for i in api.list_datasets(author=author, search='details_'):
        if i.id.casefold().find('flagged') < 0:
            yield i.id

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        fieldnames = [ x.name for x in fields(EvaluationSet) ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        for i in pool.imap_unordered(func, ls(args.author)):
            if i is not None:
                writer.writerows(map(asdict, i))
