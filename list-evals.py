import sys
import csv
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import asdict
from multiprocessing import Pool, Queue

from datasets import DownloadConfig, get_dataset_config_names
from huggingface_hub import HfApi

from mylib import Logger, EvaluationSet, AuthorModel

#
#
#
def func(incoming, outgoing, args):
    flagged = set()
    if args.flagged:
        with open(args.flagged) as fp:
            reader = csv.DictReader(fp)
            flagged.update(AuthorModel(**x) for x in reader)

    while True:
        path = incoming.get()
        Logger.info(path)

        records = []
        with TemporaryDirectory() as cache_dir:
            dc = DownloadConfig(
                cache_dir=cache_dir,
                max_retries=args.max_retries,
                disable_tqdm=True,
            )
            try:
                for i in get_dataset_config_names(path, download_config=dc):
                    if i.startswith('harness_'):
                        ev_set = EvaluationSet(path, i)
                        if flagged and ev_set.get_author_model() in flagged:
                            Logger.warning(f'Flagged model: {ev_set}')
                            continue
                        records.append(asdict(ev_set))
            except Exception as err:
                Logger.error(f'{path}: Cannot get config names ({err})')

        outgoing.put(records)

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--max-retries', type=int, default=3)
    arguments.add_argument('--flagged', type=Path)
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
