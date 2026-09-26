import sys
import csv
import json
import time
import itertools as it
import functools as ft
import statistics as st
from typing import SupportsFloat
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, fields, asdict, replace
from urllib.parse import ParseResult, urlunparse
from multiprocessing import Pool, JoinableQueue

import fsspec
import requests
import pandas as pd
from requests import HTTPError
from huggingface_hub.utils import GatedRepoError, build_hf_headers

from mylib import (
    Backoff,
    DatasetPathHandler,
    Document,
    Logger,
    QuestionBank,
    SubmissionInfo,
    SUBJECT_KEYS,
)

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
class DatasetAccessRequestor:
    _url = {
        'scheme': 'https',
        'netloc': 'huggingface.co',
    }
    _endpoint = 'ask-access'

    def __init__(self):
        self.path = DatasetPathHandler()

    def __call__(self, path):
        target = urlunparse(self.to_url(path))
        headers = build_hf_headers()

        response = requests.post(target, headers=headers)
        response.raise_for_status()

    def to_url(self, path):
        body = (self
                .path
                .relative_to(path)
                .joinpath(self._endpoint))

        kwargs = dict(self._url, path=str(body))
        for i in ParseResult._fields:
            kwargs.setdefault(i, None)

        return ParseResult(**kwargs)

class HfFileReader:
    def __init__(self, backoff, retries):
        self.ask = DatasetAccessRequestor()
        self.path = DatasetPathHandler()
        self.backoff = backoff
        self.retries = retries

    def __call__(self, target):
        url = self.path.to_string(target)
        asked = False
        last_err = None

        for delay in it.islice(self.backoff, self.retries):
            try:
                with fsspec.open(url) as fp:
                    for line in fp:
                        yield json.loads(line)
                return
            except GatedRepoError as err:
                last_err = err
                Logger.error(url)
                if not asked:
                    try:
                        self.ask(target)
                    except HTTPError as herr:
                        raise PermissionError(target) from herr
                    asked = True
                time.sleep(delay)
            except Exception as err:
                raise ConnectionError(target) from err

        raise PermissionError(target) from last_err

class SubmissionReader:
    _metrics = (
        'acc',
        'match',
    )

    def __init__(self, reader, benchmark=None):
        self.reader = reader
        self.label_key = SUBJECT_KEYS.get(benchmark)
        self.documents = []

    def __call__(self, submission):
        path = Path(submission['path'])
        for r in self.results(path):
            record = dict(submission)
            record.update(asdict(r))
            yield record

    def results(self, path):
        for line in self.reader(path):
            document = line['doc_hash']
            self.store(document, line)
            for (metric, score) in line.items():
                if any(metric.find(x) >= 0 for x in self._metrics):
                    yield Result(document, metric, score)

    def store(self, doc, info):
        label = info['doc'][self.label_key] if self.label_key else None
        self.documents.append(Document(doc, label))

#
#
#
def process(submission, hf_reader, keys, connection, args):
    Logger.info(submission['path'])

    reader = SubmissionReader(hf_reader, submission.get('benchmark'))
    try:
        df = pd.DataFrame.from_records(reader(submission))
    except (PermissionError, ConnectionError) as err:
        Logger.critical('%s: %s', type(err), err)
        return

    info = SubmissionInfo(*map(submission.get, keys))
    if not info.subject:
        info = replace(info, subject='_')

    if not df.empty:
        out = args.output.joinpath(info.to_path('.csv.gz'))
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, compression='gzip')

    qbank = QuestionBank(connection, info.benchmark, info.subject)
    qbank.write(reader.documents)

def func(tasks, args):
    hf_reader = HfFileReader(Backoff(args.backoff, 0.1), args.retries)
    keys = [ x.name for x in fields(SubmissionInfo) ]
    connection = QuestionBank.connect(args.question_bank)

    while True:
        submission = tasks.get()
        try:
            process(submission, hf_reader, keys, connection, args)
        finally:
            tasks.task_done()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--question-bank', type=Path)
    arguments.add_argument('--backoff', type=float, default=2)
    arguments.add_argument('--retries', type=int, default=3)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    tasks = JoinableQueue()
    initargs = (
        tasks,
        args,
    )

    with Pool(args.workers, func, initargs):
        reader = csv.DictReader(sys.stdin)
        for row in reader:
            tasks.put(row)
        tasks.join()
