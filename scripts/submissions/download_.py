import sys
import csv
import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, fields, asdict, astuple
from urllib.parse import ParseResult, urlunparse
from multiprocessing import Pool, JoinableQueue

import fsspec
import requests
import pandas as pd
from requests import HTTPError
from huggingface_hub.utils import GatedRepoError, build_hf_headers

from mylib import Logger, DatasetPathHandler

#
#
#
@dataclass
class Result:
    document: str
    metric: str
    score: float

@dataclass(frozen=True)
class GroupKey:
    author: str
    model: str

#
#
#
class DatasetAccessRequestor:
    _url = {
        'scheme': 'https',
        'netloc': 'huggingface.co',
    }
    _endpoint = 'ask-access'

    def __call__(self, path):
        target = urlunparse(self.to_url(path))
        headers = build_hf_headers()

        response = requests.post(target, headers=headers)
        response.raise_for_status()

    def to_url(self, path):
        body = path.parts[:3]
        path = Path(*body, 'ask-access')

        kwargs = dict(self._url, path=str(path))
        for i in ParseResult._fields:
            kwargs.setdefault(i, None)

        return ParseResult(**kwargs)

class HfFileReader:
    def __init__(self):
        self.ask = DatasetAccessRequestor()
        self.path = DatasetPathHandler()

    def __call__(self, target):
        url = self.path.to_string(target)
        for i in it.count():
            try:
                with fsspec.open(url) as fp:
                    for line in fp:
                        yield json.loads(line)
                break
            except GatedRepoError as err:
                if i:
                    raise PermissionError(target) from err
                Logger.error(url)
            except Exception as err:
                raise ConnectionError(target) from err

            try:
                self.ask(target)
            except HTTPError as err:
                raise PermissionError(target) from err

class SubmissionReader:
    _metrics = (
        'acc',
        'match',
    )

    def __init__(self):
        self.reader = HfFileReader()

    def __call__(self, df):
        for i in df.itertuples(index=False):
            path = Path(i.path)
            Logger.info(path)

            for r in self.results(path):
                record = i._asdict()
                record.update(asdict(r))
                yield record

    def results(self, path):
        for line in self.reader(path):
            document = line['doc_id']
            for (metric, score) in line.items():
                if any(metric.find(x) >= 0 for x in self._metrics):
                    yield Result(document, metric, score)

#
#
#
def func(queue, args):
    reader = SubmissionReader()

    while True:
        (key, group) = queue.get()
        Logger.warning(key)

        output = (args
                  .output
                  .joinpath(*astuple(key))
                  .with_suffix('.csv.gz'))
        output.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame.from_records(reader(group))
        df.to_csv(output, index=False, compression='gzip')

        queue.task_done()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    queue = JoinableQueue()
    initargs = (
        queue,
        args,
    )

    with Pool(args.workers, func, initargs):
        by = [ x.name for x in fields(GroupKey) ]
        df = pd.read_csv(sys.stdin, parse_dates=['date'])
        for (i, g) in df.groupby(by, sort=False):
            key = GroupKey(*i)
            queue.put((key, g))
        queue.task_done()
