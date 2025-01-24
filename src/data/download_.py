import sys
import csv
import json
import itertools as it
import functools as ft
import statistics as st
from typing import SupportsFloat
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, fields, asdict
from urllib.parse import ParseResult, urlunparse
from multiprocessing import Pool, Queue

import fsspec
import requests
import pandas as pd
from requests import HTTPError
from huggingface_hub.utils import GatedRepoError, build_hf_headers

from mylib import Logger, DatasetPathHandler, SubmissionInfo

#
# Types and functions to evaluation scores. Create new `to_float`s to
# handle special cases.
#
@ft.singledispatch
def to_float(value):
    raise TypeError('{}: {}'.format(type(value), value))

@to_float.register
def _(value: SupportsFloat): # most are float, ifeval.prompt_ is bool
    return float(value)

@to_float.register
def _(value: list): # ifeval.inst_
    return st.fmean(value)

@dataclass
class Result:
    document: str
    metric: str
    score: float

    def __post_init__(self):
        self.score = to_float(self.score)

#
#
#
@dataclass
class MySubmissionInfo(SubmissionInfo):
    def __post_init__(self):
        if not self.subject:
            self.subject = '_'

    def to_path(self, suffix):
        name = f'{self.model}{suffix}'
        return Path(self.benchmark, self.subject, self.author, name)

#
#
#
@dataclass
class QuestionBank:
    name: Path
    question: dict

    def dump(self, dst):
        output = dst.joinpath(self.name).with_suffix('.json')
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(self.question, indent=2)
            with output.open('w') as fp:
                print(data, file=fp)

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
    _document_keys = (
        'doc',
        'doc_id',
    )
    _metrics = (
        'acc',
        'match',
    )

    def __init__(self, reader):
        self.reader = reader
        self.documents = {}

    def __call__(self, submission):
        path = Path(submission['path'])
        for r in self.results(path):
            record = dict(submission)
            record.update(asdict(r))
            yield record

    def results(self, path):
        for line in self.reader(path):
            document = line['doc_hash']

            for (metric, score) in line.items():
                if any(metric.find(x) >= 0 for x in self._metrics):
                    yield Result(document, metric, score)

            self.documents[document] = {
                x: line[x] for x in self._document_keys
            }

#
#
#
def func(incoming, outgoing, args):
    hf_reader = HfFileReader()
    keys = [ x.name for x in fields(MySubmissionInfo) ]

    while True:
        submission = incoming.get()
        Logger.info(submission['path'])

        reader = SubmissionReader(hf_reader)
        try:
            df = pd.DataFrame.from_records(reader(submission))
        except (PermissionError, ConnectionError) as err:
            Logger.critical('%s: %s', type(err), err)
            outgoing.put(None)
            continue

        info = MySubmissionInfo(*map(submission.get, keys))
        out = args.output.joinpath(info.to_path('.csv.gz'))
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, compression='gzip')

        name = Path(info.benchmark, info.subject)
        outgoing.put(QuestionBank(name, reader.documents))

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--question-bank', type=Path)
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
            outgoing.put(row)
            jobs += 1

        for _ in range(jobs):
            qb = incoming.get()
            if qb is not None:
                qb.dump(args.question_bank)
