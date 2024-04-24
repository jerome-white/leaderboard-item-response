import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, fields, asdict
from multiprocessing import Pool

from datasets import DownloadConfig, get_dataset_config_names
from huggingface_hub import HfApi

from myutils import Logger, EvaluationSet

#
#
#
def func(dataset):
    Logger.info(dataset)

    with TemporaryDirectory() as cache_dir:
        dc = DownloadConfig(cache_dir=cache_dir)
        try:
            names = get_dataset_config_names(dataset, download_config=dc)
        except (ValueError, ConnectionError) as err:
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
    search = 'details_'

    for i in api.list_datasets(author=author, search=search):
        path = Path(i.id)
        assert path.name.startswith(search)
        yield path

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        fieldnames = [ x.name for x in fields(Record) ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for i in pool.imap_unordered(func, ls(args.author)):
            if i is not None:
                writer.writerows(map(asdict, i))
